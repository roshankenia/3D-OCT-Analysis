from scipy.ndimage import zoom
import matplotlib.cm as cm
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import v2
from datetime import datetime
from tqdm.auto import tqdm
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from timeSFormer.timesformer_pytorch import TimeSformer
from model import VariableAttnModelSSL
from OCT_dataloader import load_zeiss_data, load_topcon_data, load_weighted_topcon_data
from resnet import generate_model
from PIL import Image
import sys
# Import the library
import argparse
from torcheval.metrics import BinaryAUROC, BinaryPrecision, BinaryRecall
from sklearn.metrics import confusion_matrix
import yaml
import monai
from monai.losses.ssim_loss import SSIMLoss
from losses import Neg_Pearson_Loss


def save_gif(volume, save_dir, epoch, step):
    # print(volume)
    # print(np.min(volume), np.max(volume))
    imgs = [Image.fromarray(img) for img in volume]
    # duration is the number of milliseconds between frames; this is 40 frames per second
    imgs[0].save(os.path.join(save_dir, f"viz_ep_{epoch}_st_{step}.gif"),
                 save_all=True, append_images=imgs[1:], duration=50, loop=0)


def save_heatmap(inps, heatmap, type_view, save_dir, epoch, step):
    viz_map = []
    for slice in range(len(inps)):
        inp = cv2.cvtColor(inps[slice], cv2.COLOR_GRAY2BGR)
        if heatmap is not None:
            attn = cv2.applyColorMap(heatmap[slice], cv2.COLORMAP_HOT)
            combined = cv2.addWeighted(inp, 0.6, attn, 0.4, 0)
            viz_map.append(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
        else:
            viz_map.append(cv2.cvtColor(inp, cv2.COLOR_BGR2RGB))
        # if (slice+1) % 32 == 0:
        #     print('saving png')
        #     cv2.imwrite(os.path.join(
        #         attn_dir, f"{view}_num_{num}_sl_{slice+1}_t_1.png"), combined)

        # viz_map.append(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))

    imgs = [Image.fromarray(img) for img in viz_map]
    # duration is the number of milliseconds between frames; this is 40 frames per second
    imgs[0].save(os.path.join(save_dir, f"{type_view}_ep_{epoch}_st_{step}.gif"),
                 save_all=True, append_images=imgs[1:], duration=50, loop=0)


def create_heatmaps(inputs, SCARs, gradcams, save_dir, epoch, step):
    inputs_h = inputs[0][0].cpu().numpy().astype(np.uint8)
    inputs_h_above = np.transpose(inputs_h, axes=[1, 0, 2])

    save_heatmap(np.copy(inputs_h), None, 'raw_reg', save_dir, epoch, step)
    save_heatmap(np.copy(inputs_h_above), None,
                 'raw_above', save_dir, epoch, step)

    if gradcams is not None:
        gradcam = gradcams[0].detach().cpu().numpy()

        z_scale = int(inputs_h.shape[0]/gradcam.shape[0])
        y_scale = int(inputs_h.shape[1]/gradcam.shape[1])
        x_scale = int(inputs_h.shape[2]/gradcam.shape[2])
        gradcam = (zoom(gradcam, (z_scale, y_scale, x_scale))
                   * 255).astype(np.uint8)
        gradcam_above = np.transpose(gradcam, axes=[1, 0, 2])

        save_heatmap(np.copy(inputs_h), gradcam,
                     'gradcam_reg', save_dir, epoch, step)
        save_heatmap(np.copy(inputs_h_above), gradcam_above,
                     'gradcam_above', save_dir, epoch, step)

    if SCARs is not None:
        scar = SCARs[0].detach().cpu().numpy()
        z_scale = int(inputs_h.shape[0]/scar.shape[0])
        y_scale = int(inputs_h.shape[1]/scar.shape[1])
        x_scale = int(inputs_h.shape[2]/scar.shape[2])
        scar = (zoom(scar, (z_scale, y_scale, x_scale))
                * 255).astype(np.uint8)
        scar_above = np.transpose(scar, axes=[1, 0, 2])

        save_heatmap(inputs_h, scar, 'scar_reg', save_dir, epoch, step)
        save_heatmap(inputs_h_above, scar_above,
                     'scar_above', save_dir, epoch, step)


# Create the parser
parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument('--config', type=str, required=True)
# Parse the argument
args = parser.parse_args()

print(args)

# Load config
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

# exit()

seeds = config['dataset']['seeds']

use_aug = config['model']['augmentation']
use_att = config['model']['attention']
att_type = config['model']['model_params']['att_type']
aug_type = config['model']['aug_type']
if not use_att:
    att_type = 'N/A'
use_gradcam = True if config['model']['model_params']['gradcam_layer_num'] != -1 else False
use_SCAR = True if config['model']['model_params']['SCAR_layer_num'] != -1 else False

aug = 'aug' if use_aug else 'no_aug'
att = 'att' if use_att else 'no_att'
data_weight = config['dataset']['data_weight']
data_size = config['dataset']['data_size']
data_size_str = f'{data_size[0]}x{data_size[1]}x{data_size[2]}'
glaucoma_dir = config['dataset']['glaucoma_dir']
non_glaucoma_dir = config['dataset']['non_glaucoma_dir']

batch_size = config['experiment']['batch_size']
num_epochs = config['experiment']['num_epochs']
learning_rate = config['experiment']['learning_rate']
loss = config['experiment']['loss']
exp_name = config['experiment']['name']
patience = config['experiment']['patience']
exp_info = config['experiment']['info']
use_ssl = config['experiment']['use_ssl']
ssl_weight = config['experiment']['ssl_weight']
ssl_loss_name = config['experiment']['ssl_loss_name']


device = torch.device(config['experiment']['device'])

root_dir = './'+config['dataset']['provider']+'_Results/'
if not os.path.isdir(root_dir):
    os.makedirs(root_dir)

if use_gradcam:
    g_layer = config['model']['model_params']['gradcam_layer_num']
    gradcam_text = f'gradcam_{g_layer}'
else:
    gradcam_text = 'no_gradcam'
if use_SCAR:
    s_layer = config['model']['model_params']['SCAR_layer_num']
    scar_text = f'scar_{s_layer}'
else:
    scar_text = 'no_scar'
if use_ssl:
    ssl_text = f'ssl_{ssl_loss_name}_{str(ssl_weight).replace(".", "_")}'
else:
    ssl_text = 'no_ssl'
# exp_dir = os.path.join(
#     root_dir, exp_name, f'{exp_info}_{gradcam_text}_{scar_text}_{data_weight}_{aug}_{att}_{data_size_str}')
exp_dir = os.path.join(
    root_dir, exp_name, f'{exp_info}_{ssl_text}_{gradcam_text}_{scar_text}_{aug}_{att}')
if not os.path.isdir(exp_dir):
    os.makedirs(exp_dir)


accs = []
specs = []
sens = []
aurocs = []
for seed in seeds:
    main_dir = os.path.join(exp_dir, f'trial_{seed}')
    if not os.path.isdir(main_dir):
        os.makedirs(main_dir)
    model_dir = os.path.join(main_dir, 'models')
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    viz_dir = os.path.join(main_dir, 'data')
    if not os.path.isdir(viz_dir):
        os.makedirs(viz_dir)

    # set stdout and stderr
    sys.stdout = open(os.path.join(main_dir, 'results.log'), 'w')
    sys.stderr = sys.stdout

    print(f'Starting trial with seed {seed}')
    print(f'Config: {config}')

    # save yaml for future reference
    with open(os.path.join(main_dir, 'config.yml'), 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    if 'Zeiss' in main_dir:
        training_loader = load_zeiss_data(
            batch_size=batch_size, att_type=att_type, dataset_type='train', shuffle=True, augment=use_aug, seed=seed)
        val_loader = load_zeiss_data(
            batch_size=batch_size, att_type=att_type, dataset_type='val', seed=seed)
    else:
        if data_weight != 'same':
            training_loader = load_weighted_topcon_data(glaucoma_dir=glaucoma_dir, non_glaucoma_dir=non_glaucoma_dir, data_size=data_size, att_type=att_type,
                                                        batch_size=batch_size, dataset_type='train', shuffle=True, augment=use_aug, weighting=data_weight, seed=seed, aug_type=aug_type)
        else:

            training_loader = load_topcon_data(glaucoma_dir=glaucoma_dir, non_glaucoma_dir=non_glaucoma_dir, data_size=data_size, att_type=att_type,
                                               batch_size=batch_size, dataset_type='train', shuffle=True, augment=use_aug, weighting=data_weight, seed=seed, aug_type=aug_type)

        val_loader = load_topcon_data(glaucoma_dir=glaucoma_dir, non_glaucoma_dir=non_glaucoma_dir, data_size=data_size, att_type=att_type, batch_size=batch_size, dataset_type='val',
                                      weighting=data_weight, seed=seed, aug_type=aug_type)

    if use_att:
        model_params = config['model']['model_params']
        model = VariableAttnModelSSL(
            input_size=data_size, model_params=model_params)
    # model = generate_model(model_depth=50, n_input_channels=1, n_classes=1)
    print(model)
    model.to(device)

    criterion = torch.nn.BCELoss()

    if use_ssl:
        if ssl_loss_name == 'MSE':
            ssl_criterion = torch.nn.MSELoss(reduction='mean')
        elif ssl_loss_name == 'KL':
            ssl_criterion = torch.nn.KLDivLoss()
        elif ssl_loss_name == 'SSIM':
            ssl_criterion = SSIMLoss(spatial_dims=2)
        elif ssl_loss_name == 'Pearson':
            ssl_criterion = Neg_Pearson_Loss()
    ssl_loss = 0
    gradcams = None

    # Optimizers specified in the torch.optim package
    optimizer = torch.optim.NAdam(model.parameters(), lr=learning_rate)

    best_val_loss = 100000.
    last_update_count = 0

    training_losses = []
    validation_losses = []
    stop = False
    for epoch in range(1, num_epochs+1):
        with tqdm(training_loader, unit="batch") as tepoch:
            acc_loss = 0.
            step = 0
            for inputs, labels in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                inputs, labels = inputs.to(device), labels.to(device)
                outputs, SCARs = model(inputs)

                # forward + backward + optimize
                optimizer.zero_grad()         # Clear old gradients

                classification_loss = criterion(outputs, labels)

                if use_gradcam:
                    # GradCAM computation
                    optimizer.zero_grad()

                    predicted_classes = torch.argmax(outputs, dim=1)
                    grad_outputs = torch.zeros_like(outputs)
                    grad_outputs[torch.arange(
                        outputs.size(0)), predicted_classes] = 1.0
                    outputs.backward(gradient=grad_outputs, retain_graph=True)

                    # Compute Grad-CAM heatmaps
                    gradcams = model.compute_gradcam_batch()

                if use_ssl:
                    # print(f'gradcams: {gradcams}')
                    # print(
                    #     f'gradcams max/min: {torch.max(gradcams)}/{torch.min(gradcams)}')

                    # print(f'SCARs: {SCARs}')
                    # print(
                    #     f'SCARs max/min: {torch.max(SCARs)}/{torch.min(SCARs)}')

                    ssl_loss = ssl_criterion(gradcams, SCARs)
                    optimizer.zero_grad()
                    loss = (1-ssl_weight)*classification_loss + \
                        ssl_weight*ssl_loss
                else:
                    optimizer.zero_grad()
                    loss = classification_loss

                loss.backward()
                optimizer.step()

                # create heatmaps
                if epoch % 5 == 0 and step == 1:
                    print(
                        f'classification loss:{classification_loss} ssl loss:{ssl_loss}')
                    print(
                        f'inputs max/min/shape: {inputs.shape} / {torch.max(inputs)} / {torch.min(inputs)}')
                    if SCARs is not None:
                        print(
                            f'SCARs max/min/shape: {SCARs.shape} / {torch.max(SCARs)} / {torch.min(SCARs)}')
                    if gradcams is not None:
                        print(
                            f'gradcams max/min/shape: {gradcams.shape} / {torch.max(gradcams)} / {torch.min(gradcams)}')
                    create_heatmaps(inputs, SCARs, gradcams,
                                    viz_dir, epoch, step)

                acc_loss += loss.item()
                step += 1

                tepoch.set_postfix(loss=acc_loss/step)

        training_losses.append(acc_loss/step)

        # validate after every epoch
        with torch.no_grad():
            bin_preds = []
            bin_labels = []
            with tqdm(val_loader, unit="batch") as tepoch:
                acc_loss = 0.
                step = 0
                for inputs, labels in tepoch:
                    tepoch.set_description(f"Validation Epoch {epoch}")

                    inputs, labels = inputs.to(device), labels.to(device)

                    # forward + backward + optimize
                    outputs, _ = model(inputs)
                    loss = criterion(outputs, labels)

                    acc_loss += loss.item()
                    step += 1

                    tepoch.set_postfix(loss=acc_loss/step)

                    preds = torch.round(outputs)
                    bin_preds.extend(outputs[:, 1].cpu().numpy())
                    bin_labels.extend(labels[:, 1].cpu().numpy())

                total_loss = acc_loss/step
                if total_loss < best_val_loss:
                    last_update_count = 0
                    print(
                        f'New best validation loss {total_loss} vs. {best_val_loss} found in epoch {epoch}, saving model...')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_val_loss,
                    }, os.path.join(model_dir, f'./best_model.pt'))
                    # update best loss
                    best_val_loss = total_loss

                    # calculate metrics
                    bin_preds = torch.round(torch.tensor(bin_preds)).long()
                    bin_labels = torch.tensor(bin_labels).long()

                    tn, fp, fn, tp = confusion_matrix(
                        bin_labels.numpy(), bin_preds.numpy()).ravel()
                    recall = tp/(tp+fn)
                    precision = tp/(tp+fp)
                    specificity = tn / (tn+fp)
                    sensitivity = tp/(tp+fn)
                    accuracy = (tp+tn)/(tp+fp+tn+fn)
                    print(f'\tPrecision: {precision}')
                    print(f'\tRecall: {recall}')
                    print(f'\tSpecificity: {specificity}')
                    print(f'\tSensitivity: {sensitivity}')
                    print(f'\tAccuracy: {accuracy}')
                else:
                    last_update_count += 1
                validation_losses.append(total_loss)

                if last_update_count >= patience:
                    print(f'Stopping early as patience threshold reached.')
                    stop = True
        if stop:
            break

    print(f'Finished Training after {epoch} epochs')

    # make plot
    plt.clf()
    x = list(range(1, epoch+1))
    plt.plot(x, training_losses, label='Training Loss')
    plt.plot(x, validation_losses, label='Validation Loss')
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.savefig(os.path.join(main_dir, f'./losses.png'))
    plt.close()

    # TESTING
    if 'Zeiss' in main_dir:
        test_loader = load_zeiss_data(
            batch_size=batch_size, dataset_type='test', att_type=att_type, seed=seed)
    else:
        test_loader = load_topcon_data(glaucoma_dir=glaucoma_dir, non_glaucoma_dir=non_glaucoma_dir, data_size=data_size, att_type=att_type,
                                       batch_size=batch_size, dataset_type='test', weighting=data_weight, seed=seed, aug_type=aug_type)

    model_name = os.path.join(model_dir, 'best_model.pt')
    checkpoint = torch.load(model_name, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    model.to(device)

    auc = BinaryAUROC()
    prec = BinaryPrecision()
    rec = BinaryRecall()

    # test
    bin_preds = []
    bin_labels = []
    with torch.no_grad():
        count = 0
        length = 0
        with tqdm(test_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Testing model")
            for inputs, labels in tepoch:

                inputs, labels = inputs.to(device), labels.to(device)

                # forward
                outputs, _ = model(inputs)
                preds = torch.round(outputs)
                for p in range(len(preds)):
                    print(
                        f'Pred {preds[p].cpu().numpy()} and Label {labels[p].cpu().numpy()}')
                print()
                correct = (torch.argmax(preds, dim=1) ==
                           torch.argmax(labels, dim=1)).sum().item()
                count += correct
                length += len(preds)

                # last column of outputs is "probability" of glaucoma
                bin_preds.extend(outputs[:, 1].cpu().numpy())
                bin_labels.extend(labels[:, 1].cpu().numpy())
            print(f'Test accuracy: {count / length}', flush=True)
    bin_preds = torch.round(torch.tensor(bin_preds)).long()
    bin_labels = torch.tensor(bin_labels).long()
    print()
    print(f'preds: {bin_preds} \nlabels: {bin_labels}', flush=True)
    auc.update(bin_preds, bin_labels)
    auroc = auc.compute()
    print(f'Binary AUROC: {auroc}', flush=True)
    prec.update(bin_preds, bin_labels)
    print(f'Binary Precision: {prec.compute()}', flush=True)
    rec.update(bin_preds, bin_labels)
    print(f'Binary Recall: {rec.compute()}', flush=True)

    tn, fp, fn, tp = confusion_matrix(
        bin_labels.numpy(), bin_preds.numpy()).ravel()
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    specificity = tn / (tn+fp)
    sensitivity = tp/(tp+fn)
    accuracy = (tp+tn)/(tp+fp+tn+fn)
    print(f'Precision: {precision}', flush=True)
    print(f'Recall: {recall}', flush=True)
    print(f'Specificity: {specificity}', flush=True)
    print(f'Sensitivity: {sensitivity}', flush=True)
    print(f'Accuracy: {accuracy}', flush=True)

    accs.append(accuracy)
    specs.append(specificity)
    sens.append(sensitivity)
    aurocs.append(auroc)

# log entire trial results to main folder
sys.stdout = open(os.path.join(exp_dir, 'results.log'), 'w')
sys.stderr = sys.stdout

avg_acc = np.round(np.mean(accs), decimals=4)
avg_spec = np.round(np.mean(specs), decimals=4)
avg_sens = np.round(np.mean(sens), decimals=4)
avg_auroc = np.round(np.mean(aurocs), decimals=4)

std_acc = np.round(np.std(accs), decimals=4)
std_spec = np.round(np.std(specs), decimals=4)
std_sens = np.round(np.std(sens), decimals=4)
std_auroc = np.round(np.std(aurocs), decimals=4)

print(f'Average metrics over seeds {seeds}:')
print(
    f'Average \u00B1 Standard Deviation for Test Accuracy: {avg_acc} \u00B1 {std_acc} using: {accs}')
print(
    f'Average \u00B1 Standard Deviation for Test Specificity: {avg_spec} \u00B1 {std_spec} using: {specs}')
print(
    f'Average \u00B1 Standard Deviation for Test Sensitivity: {avg_sens} \u00B1 {std_sens} using: {sens}')
print(
    f'Average \u00B1 Standard Deviation for Test AUROC: {avg_auroc} \u00B1 {std_auroc} using: {aurocs}')
