import os
import torch.optim as optim
import torch
from Dataset import Dataset
from SimCLR import SimCLR
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import cam

# Train classifier
def finetune(train_loader,testloader=None,model_path='ModelSimclr.pt'):
    #Retrieve model
    simclr = SimCLR(train_loader)
    simclr.model.load_state_dict(torch.load(model_path))

    simclr.model.fc = nn.Identity()
    for param in simclr.model.parameters():
        param.requires_grad = False

    model = simclr.model

    model.fc = nn.Linear(512, 2)

    model.to(simclr.device)
    print(simclr.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
    device = simclr.device
    for epoch in range(1):
        running_loss = 0.0
        model.train()
        correct = 0
        total = 0
        for i, ((inputs, _), labels) in enumerate(train_loader):
            # zero the parameter gradients
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()
            if i % 20 == 0:  # print every 20 mini-batches
                print('[%d, %5d] loss: %.5f' %
                      (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0
                print('Accuracy of the network on the trainin images: %d %%' % (
                        100 * correct / total))
    print('Finished Training')
    torch.save(model.state_dict(), 'classifier.pt')



# Evaluate
def test_model(set='C1'):

    test = Dataset(set, setting='train', sim=False, original=True)
    testloader = torch.utils.data.DataLoader(test, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
    simclr = SimCLR(testloader)
    model = simclr.model
    model.fc = nn.Linear(512, 2)
    model.load_state_dict(torch.load('classifier.pt'))
    verbose = False
    device = simclr.device
    correct = 0
    total = 0
    model.to(device)
    model.eval()
    false_positives = 0
    total_negatives = 0
    false_negatives = 0
    true_positives = 0
    true_negatives = 0
    print(len(testloader.dataset))

    with torch.no_grad():
        for i, data in enumerate(testloader):
            (images, augm, original), labels, path = data

            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            plt.tick_params(
                axis='both',  # changes apply to the x-axis
                # which='both',      # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,
                labelleft=False,
                left=False,
            )
            if verbose:
                plt.imshow(images.cpu().detach().squeeze().numpy().transpose(1, 2, 0) / 255)
                score = nn.Softmax()(outputs).max().item()

                plt.show()
                plt.tick_params(
                    axis='both',  # changes apply to the x-axis
                    # which='both',      # both major and minor ticks are affected
                    bottom=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    labelbottom=False,
                    labelleft=False,
                    left=False,
                )
                plt.imshow(augm.cpu().detach().squeeze().squeeze().numpy().transpose(1, 2, 0) / 255)
                plt.show()

            total_negatives += (labels == 0).sum().item()
            false_positives += (predicted[labels == 0] == 1).sum().item()
            false_negatives += (predicted[labels == 1] == 0).sum().item()
            true_positives += (predicted[labels == 1] == 1).sum().item()
            true_negatives += (predicted[labels == 0] == 0).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (
            100 * correct / total))

    print('Finished Training')
    print('Correct : ', correct)
    print('Total : ', total)
    print('False Positives : ', false_positives)
    print('True Positives: ', true_positives)
    print('False Negatives : ', false_negatives)
    print('True Negatives : ', true_negatives)


if __name__ == '__main__':

    data_dir = 'S1/Train'
    test_dir = 'S1/Test'

    #For SimcLR Training sim = True. For finetuning sim = False

    train_dataset = Dataset(data_dir,setting='train',sim=True)
    val_dataset =  Dataset(test_dir,setting='test',original=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=112, shuffle=True, num_workers=1, drop_last = True)
    testloader = torch.utils.data.DataLoader(val_dataset, batch_size=112, shuffle=True, num_workers=1, drop_last = False)

    #Begin SimCLR Training
    simclr = SimCLR(train_loader,epochs=200,batch_size=112)
    simclr.train()

    torch.save(simclr.model.state_dict(), 'ModelSimclr.pt')

    #For SimcLR Training sim = True. For finetuning sim = False
    train_dataset = Dataset(data_dir,setting='train',sim=False)
    val_dataset =  Dataset(test_dir,setting='test',original=False)
    torch.manual_seed(999)
    print('Training Classifier')
    finetune(train_loader,model_path='ModelSimclr.pt')
    print('Evaluation')
    test_model()
    cam('C1/1/iceland_d_20201020_20201119.geo.diff1_2.png')
