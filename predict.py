import torch
import matplotlib.pyplot as plt
import model as md
import prepare_data as data

model_name = 'model_0.9798385305191156.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = md.get_model()
model.load_state_dict(torch.load('./models/' + model_name))
model.to(device)
model.eval()

# Préparation des données pour la prédiction
_, valid_set = data.get_data()
valid_loader = data.get_validloader(valid_set)


def predict(max_samples=100):
    nb_error = 0
    nb_correct = 0

    images_valid_list = []
    predicted_valid_list = []
    labels_valid_list = []

    images_error_list = []
    predicted_error_list = []
    labels_error_list = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(valid_loader):

            if i > max_samples:
                break

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            if predicted != labels:
                nb_error += 1

                images_error_list.append(images[0].cpu())
                predicted_error_list.append(predicted.item())
                labels_error_list.append(labels.item())
            else:
                nb_correct += 1

                images_valid_list.append(images[0].cpu())
                predicted_valid_list.append(predicted.item())
                labels_valid_list.append(labels.item())

        accuracy = nb_correct / (nb_correct + nb_error)
        print(f"Number of errors: {nb_error}")
        print(f"Number of correct: {nb_correct}")
        print(f"Accuracy: {accuracy:.5f}")

    return (images_valid_list, predicted_valid_list, labels_valid_list,
            images_error_list, predicted_error_list, labels_error_list)


def imshow(img, predicted, label):
    img = img.cpu().numpy().squeeze()
    plt.imshow(img, cmap='gray')
    plt.title(f'Predicted: {predicted}, Label: {label}')
    plt.show()


def plot_images(images, predicted, labels, images_error, predicted_error, labels_error):
    n_valid = len(images)
    n_error = len(images_error)
    cols = 5
    rows_valid = (n_valid + cols - 1) // cols  # Calcul du nombre de lignes nécessaires
    rows_error = (n_error + cols - 1) // cols
    plt.figure(figsize=(15, 3 * rows_valid))
    
    for i in range(n_valid):
        plt.subplot(rows_valid, cols, i + 1)
        img = images[i].numpy().squeeze()
        plt.imshow(img, cmap='gray')
        plt.title(f'Pred: {predicted[i]}, Label: {labels[i]}')
        plt.axis('off')

    plt.savefig(f'./output/predictions_valid_{model_name}.png')  # Sauvegarder l'image au lieu de l'afficher

    if len(images_error) > 0:
        plt.figure(figsize=(15, 3 * rows_error))

        for i in range(n_error):
            plt.subplot(rows_error, cols, i + 1)
            img = images_error[i].numpy().squeeze()
            plt.imshow(img, cmap='gray')
            plt.title(f'Pred: {predicted_error[i]}, Label: {labels_error[i]}')
            plt.axis('off')

        plt.savefig(f'./output/predictions_errors_{model_name}.png')
    else :
        print("No images error")

    plt.close()


(images_valid_list, predicted_valid_list, labels_valid_list,
 images_error_list, predicted_error_list, labels_error_list) = predict()

plot_images(images_valid_list, predicted_valid_list, labels_valid_list,
            images_error_list, predicted_error_list, labels_error_list)
