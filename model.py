import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
import numpy as np
from transformer_block import TransformerBlock, EPABlock, CrossAttentionBlockSIMN, CrossAttentionBlockSIIS, SplitCrossAttentionBlockSIIS, MultiSplitCrossAttentionBlockSIIS, CrossAttentionBlockMNNM
import monai


def make_block(in_channels, out_channels, kernel_size, stride, padding, num_features):
    # make a 3D convolution
    block = nn.Sequential(nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=kernel_size, stride=stride, padding=padding), nn.BatchNorm3d(num_features=num_features), nn.ReLU())
    return block


def make_conv_block(in_channels, out_channels, kernel_size, stride, padding):
    # make a 3D convolution
    block = nn.Sequential(nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=kernel_size, stride=stride, padding=padding), nn.BatchNorm3d(num_features=out_channels), nn.ReLU())
    return block


def make_block_with_pool(in_channels, out_channels, kernel_size, stride, padding, num_features):
    # make a 3D convolution
    block = nn.Sequential(nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=kernel_size, stride=stride, padding=padding), nn.BatchNorm3d(num_features=num_features), nn.ReLU(), nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
    return block


def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a


class New_OCT_Model(nn.Module):
    def __init__(self, input_size):
        super(New_OCT_Model, self).__init__()
        self.conv_block1 = make_block(
            in_channels=1, out_channels=32, kernel_size=7, stride=2, padding=3, num_features=32)
        self.conv_block2 = make_block(
            in_channels=32, out_channels=32, kernel_size=5, stride=1, padding='same', num_features=32)
        self.conv_block3 = make_block(
            in_channels=32, out_channels=32, kernel_size=3, stride=1, padding='same', num_features=32)
        self.conv_block4 = make_block(
            in_channels=32, out_channels=32, kernel_size=3, stride=1, padding='same', num_features=32)
        self.conv_block5 = make_block(
            in_channels=32, out_channels=32, kernel_size=3, stride=1, padding='same', num_features=32)

        print(f'NOT using attention in OCT model')

        stride = np.array(input_size)/2
        stride = stride.astype(np.int64)

        kernel_size = int(np.min(input_size)/8)
        self.GAP = nn.AvgPool3d(kernel_size=kernel_size, stride=tuple(stride))
        self.dense = nn.Linear(in_features=32, out_features=2)

        self.Softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        # print('1', x.shape)
        x = self.conv_block1(x)
        # print('2', x.shape)
        x = self.conv_block2(x)
        # print('3', x.shape)
        x = self.conv_block3(x)
        # print('4', x.shape)
        x = self.conv_block4(x)
        # print('5', x.shape)
        x = self.conv_block5(x)
        # print('6', x.shape)
        # print('7', x.shape)
        x = torch.squeeze(self.GAP(x))
        # print('8', x.shape)
        x = self.dense(x)
        # make sure 2D tensor
        x = x.view(batch_size, -1)
        # print('9', x.shape)
        x = self.Softmax(x)

        return x


class VariableAttnModel(nn.Module):
    def __init__(self, input_size, model_params):
        super(VariableAttnModel, self).__init__()
        att_type = model_params['att_type']
        att_ind = model_params['att_ind']
        max_pool_ind = model_params['max_pool_ind']
        print(
            f'Using variable {att_type} attention in OCT model at ind {att_ind}')

        self.layers = nn.ModuleList()

        for i in range(0, model_params['num_conv_layers']):
            self.layers.append(make_conv_block(in_channels=model_params['conv_in_channels'][i], out_channels=model_params['conv_out_channels'][i],
                               kernel_size=model_params['conv_kernel_size'][i], stride=model_params['conv_stride'][i], padding=model_params['conv_padding'][i]))
            if i+1 in att_ind:
                print(f'Using attention + max_pool after conv layer {i+1}')
                if att_type == 'CrossSIIS':
                    self.layers.append(nn.Sequential(CrossAttentionBlockSIIS(hidden_size=model_params['conv_out_channels'][i], num_heads=4,
                                                                             dropout_rate=0.15), torch.nn.MaxPool3d(kernel_size=(2, 2, 2))))
                elif att_type == 'MultiSplitCrossSIIS':
                    self.layers.append(nn.Sequential(MultiSplitCrossAttentionBlockSIIS(hidden_size=model_params['conv_out_channels'][i], num_heads=4,
                                                                                       dropout_rate=0.15), torch.nn.MaxPool3d(kernel_size=(2, 2, 2))))
                elif att_type == 'CrossMNNM':
                    self.layers.append(nn.Sequential(CrossAttentionBlockMNNM(hidden_size=model_params['conv_out_channels'][i], num_heads=4,
                                                                             dropout_rate=0.15), torch.nn.MaxPool3d(kernel_size=(2, 2, 2))))
            if i+1 in max_pool_ind:
                print(f'Using max_pool after conv layer {i+1}')
                self.layers.append(torch.nn.MaxPool3d(kernel_size=(2, 2, 2)))

        conv_down = np.sum(
            model_params['conv_stride'])-len(model_params['conv_stride'])
        stride = np.array(input_size) / np.power(2,
                                                 len(att_ind)+len(max_pool_ind)+conv_down)
        stride = stride.astype(np.int64)
        kernel_size = int(np.min(stride))
        self.GAP = nn.AvgPool3d(kernel_size=kernel_size, stride=tuple(stride))
        self.dense = nn.Linear(
            in_features=model_params['conv_out_channels'][-1], out_features=2)

        self.Softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # print(f'shape after layer {i}: {x.shape}')
        x = torch.squeeze(self.GAP(x))
        # print('8', x.shape)
        x = self.dense(x)
        # make sure 2D tensor
        x = x.view(batch_size, -1)
        # print('9', x.shape)
        x = self.Softmax(x)

        return x


class VariableAttnModelSSL(nn.Module):
    def __init__(self, input_size, model_params):
        super(VariableAttnModelSSL, self).__init__()
        att_type = model_params['att_type']
        att_ind = model_params['att_ind']
        max_pool_ind = model_params['max_pool_ind']
        print(
            f'Using variable {att_type} attention in OCT model at ind {att_ind}')

        self.layers = nn.ModuleList()

        self.gradcam_layer_num = model_params['gradcam_layer_num'] - 1
        self.SCAR_layer_num = model_params['SCAR_layer_num']
        self.SCAR_layer = None

        def forward_hook(module, input, output):
            self.hooked_activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        for i in range(0, model_params['num_conv_layers']):
            self.layers.append(make_conv_block(in_channels=model_params['conv_in_channels'][i], out_channels=model_params['conv_out_channels'][i],
                               kernel_size=model_params['conv_kernel_size'][i], stride=model_params['conv_stride'][i], padding=model_params['conv_padding'][i]))
            if self.gradcam_layer_num == i:
                # Convolution within sequential block
                conv3d_layer = self.layers[-1][0]
                conv3d_layer.register_forward_hook(forward_hook)
                conv3d_layer.register_backward_hook(backward_hook)

            if i+1 in att_ind:
                print(f'Using attention + max_pool after conv layer {i+1}')
                if att_type == 'CrossSIIS':
                    self.layers.append(nn.Sequential(CrossAttentionBlockSIIS(hidden_size=model_params['conv_out_channels'][i], num_heads=4,
                                                                             dropout_rate=0.15), torch.nn.MaxPool3d(kernel_size=(2, 2, 2))))
                elif att_type == 'MultiSplitCrossSIIS':
                    self.layers.append(nn.Sequential(MultiSplitCrossAttentionBlockSIIS(hidden_size=model_params['conv_out_channels'][i], num_heads=4,
                                                                                       dropout_rate=0.15), torch.nn.MaxPool3d(kernel_size=(2, 2, 2))))
                elif att_type == 'CrossMNNM':
                    self.layers.append(nn.Sequential(CrossAttentionBlockMNNM(hidden_size=model_params['conv_out_channels'][i], num_heads=4,
                                                                             dropout_rate=0.15), torch.nn.MaxPool3d(kernel_size=(2, 2, 2))))
                if i+1 == self.SCAR_layer_num:
                    self.SCAR_layer = self.layers[-1]
            if i+1 in max_pool_ind:
                print(f'Using max_pool after conv layer {i+1}')
                self.layers.append(torch.nn.MaxPool3d(kernel_size=(2, 2, 2)))

        conv_down = np.sum(
            model_params['conv_stride'])-len(model_params['conv_stride'])
        stride = np.array(input_size) / np.power(2,
                                                 len(att_ind)+len(max_pool_ind)+conv_down)
        stride = stride.astype(np.int64)

        # stride = np.array(input_size)/2
        # stride = stride.astype(np.int64)
        # kernel_size = 32
        # if stride[0] == 16:
        #     kernel_size = 16
        # kernel_size = int(np.min(input_size)/8)
        kernel_size = int(np.min(stride))
        self.GAP = nn.AvgPool3d(kernel_size=kernel_size, stride=tuple(stride))
        self.dense = nn.Linear(
            in_features=model_params['conv_out_channels'][-1], out_features=2)

        self.Softmax = torch.nn.Softmax(dim=1)
        # Grad-CAM variables
        self.hooked_activations = None
        self.gradients = None

    def forward(self, x):
        batch_size = x.shape[0]
        attn_return = None
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # print(f'shape after layer {i}: {x.shape} {layer}')
            if layer == self.SCAR_layer:
                attn_return = x
        x = torch.squeeze(self.GAP(x))
        # print('8', x.shape)
        x = self.dense(x)
        # make sure 2D tensor
        x = x.view(batch_size, -1)
        # print('9', x.shape)
        x = self.Softmax(x)

        if self.SCAR_layer is not None:
            # Sum across channels to generate heatmap
            # attn_return = torch.sum(attn_return, dim=1)
            attn_return = torch.mean(attn_return, dim=1)
            # attn_return = torch.logsumexp(attn_return, dim=1)

            # ReLU and normalization
            attn_return = torch.nn.functional.relu(attn_return)
            max_vals = torch.amax(attn_return, dim=(1, 2, 3), keepdim=True)
            # attn_return = torch.nn.functional.normalize(attn_return)
            attn_return = attn_return / (max_vals + 1e-5)

        return x, attn_return

    def compute_gradcam_batch(self):
        """
        Compute Grad-CAM heatmaps for the entire batch.

        Returns:
            list of numpy.ndarray: Grad-CAM heatmaps for each input in the batch.
        """
        if self.hooked_activations is None or self.gradients is None:
            raise ValueError(
                "Forward and backward hooks have not captured data.")

        # Global average pooling over gradients
        # Shape: (batch_size, num_channels)
        pooled_grads = torch.mean(self.gradients, dim=(2, 3, 4))

        # Expand dimensions to match activation shape
        pooled_grads = pooled_grads[:, :, None, None, None]

        # Weight activations by pooled gradients
        weighted_activations = self.hooked_activations * pooled_grads

        # Sum across channels to generate heatmap
        # Summing over channels
        heatmaps = torch.sum(weighted_activations, dim=1)

        # Apply ReLU and normalization
        heatmaps = torch.nn.functional.relu(heatmaps)
        max_vals = torch.amax(heatmaps, dim=(1, 2, 3), keepdim=True)
        heatmaps = heatmaps / (max_vals + 1e-5)

        return heatmaps

    # def compute_gradcam(self):
    #     """
    #     Compute Grad-CAM heatmaps for the entire batch.

    #     Returns:
    #         list of numpy.ndarray: Grad-CAM heatmaps for each input in the batch.
    #     """
    #     if self.hooked_activations is None or self.gradients is None:
    #         raise ValueError(
    #             "Forward and backward hooks have not captured data.")

    #     batch_size = self.hooked_activations.shape[0]
    #     heatmaps = []

    #     for i in range(batch_size):
    #         # Global average pooling on gradients for the specific input
    #         pooled_grads = torch.mean(self.gradients[i], dim=(1, 2, 3))

    #         # Weight activations by pooled gradients
    #         weighted_activations = self.hooked_activations[i] * \
    #             pooled_grads[:, None, None, None]

    #         # Sum across channels to generate heatmap
    #         heatmap = torch.sum(weighted_activations, dim=0)

    #         # Apply ReLU and normalization
    #         heatmap = torch.nn.functional.relu(heatmap)
    #         heatmap = heatmap/torch.max(heatmap) + 1e-5

    #         heatmaps.append(heatmap)

    #     return heatmaps


# Create DenseNet121, CrossEntropyLoss and Adam optimizer
# model = monai.networks.nets.DenseNet121(
#     spatial_dims=3, in_channels=1, out_channels=2).to(device)
# print(model)
# out = model(rand)
# print(out.shape)
# print(out)

# from monai.networks.nets import AutoEncoder

# # 3 layers each down/up sampling their inputs by a factor 2 with no intermediate layer
# model = AutoEncoder(
#     spatial_dims=3,
#     in_channels=1,
#     out_channels=2,
#     channels=(2, 4, 8),
#     strides=(2, 2, 2)
# ).to(device)
# print(model)
# out = model(rand)
# print(out.shape)
# # print(out)
# model_params = {
#     'att_type': 'CrossSIIS',
#     # 'att_type': 'MultiSplitCrossSIIS',
#     # 'att_type' : 'CrossMNNM',
#     'num_conv_layers': 8,
#     'conv_kernel_size': [3, 3, 3, 3, 3, 3, 3, 3],
#     'conv_in_channels': [1, 16, 16, 32, 32, 32, 32, 64],
#     'conv_out_channels': [16, 16, 32, 32, 32, 32, 64, 128],
#     'conv_stride': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     'conv_padding':
#     ["same", "same", "same", "same", "same", "same", "same", "same"],
#         'att_ind': [2, 5],
#         'max_pool_ind': [7],
#         'gradcam_layer_num': 8,
#         'SCAR_layer_num': 2,
# }

# model_params = {
#     'att_type': "CrossSIIS",
#     'num_conv_layers': 5,
#     'conv_kernel_size': [7, 5, 3, 3, 3],
#     'conv_in_channels': [1, 32, 32, 32, 32],
#     'conv_out_channels': [32, 32, 32, 32, 32],
#     'conv_stride': [2, 1, 1, 1, 1],
#     'conv_padding': [3, "same", "same", "same", "same"],
#     'att_ind': [2, 4],
#     'max_pool_ind': [],
#     'gradcam_layer_num': 5,
#     'SCAR_layer_num': 4,
# }

# device = torch.device("cuda:1")

# rand = torch.rand(8, 1, 128, 192, 112).to(device)
# # model = VariableAttnModel(
# #     (128, 192, 112), model_params=model_params).to(device)
# model = VariableAttnModelSSL(
#     (128, 192, 112), model_params=model_params).to(device)
# # model = Cross_Attn_OCT_Model(input_size=[128, 192, 112], att_type="EPA").to(device)

# # print(model)
# # model = Cross_Attn_OCT_Model(input_size=(128, 192, 112), att_type="CrossSIIS").to(device)
# # print(model)
# out, _ = model(rand)
# print(out.shape)
# out, attn = model(rand)
# print(out.shape)
# print(attn.shape)

# # GradCAM computation
# predicted_classes = torch.argmax(out, dim=1)
# grad_outputs = torch.zeros_like(out)
# grad_outputs[torch.arange(
#     out.size(0)), predicted_classes] = 1.0
# out.backward(gradient=grad_outputs)

# # Compute Grad-CAM heatmaps
# batch_heatmaps = model.compute_gradcam_batch()
# print(batch_heatmaps[0].shape)
