from src.Config.Args import args
import torch


def train(model, train_loader, optimizer, criterion, epoch, transform):
    # Set the model to training mode
    model.train()

    # Retain a history of losses
    all_losses = []

    # Iterate through the data loarder and train the model
    for batch_idx, entry in enumerate(train_loader):
        # Send the data to the available GPU
        images = entry['image'].to(args.device)

        # Preprocess the images
        images = transform(images)
        
        # Reset the gradients back to zero before calculating new loss
        optimizer.zero_grad()

        # Compute the output of the model
        output = model(images)

        # Calculate the loss accordingly
        loss = criterion(output, images)

        # Add the current loss to the history
        all_losses.append(loss.cpu())

        # Use backpropagation to calculate the new gradients
        loss.backward()

        # Use a custom optimizer to adjuts the parameters of the model
        optimizer.step()

        # Show updates regularly
        if batch_idx % args.log_interval == 0 or batch_idx == len(train_loader) - 1:
            print(f'Train Epoch: {epoch} ', end='')
            print(f'[{batch_idx * images.shape[0]}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]', end='')
            print(f'\tLoss: {loss.item()}')

    # Return the losses of this epoch
    return torch.tensor(all_losses).mean()


def test(model, test_loader, criterion, epoch, transform):    
    # Set the model to training mode
    model.eval()

    # Retain a history of losses
    all_losses = []
    
    # Don't store activations
    with torch.no_grad():
        # Iterate through the data loarder and train the model
        for batch_idx, entry in enumerate(test_loader):
            # Send the data to the available GPU
            images = entry['image'].to(args.device)

            # Preprocess the images
            images = transform(images)

            # Compute the output of the model
            output = model(images)

            # Calculate the loss accordingly
            loss = criterion(output, images)

            # Add the current loss to the history
            all_losses.append(loss.cpu())

            # Show updates regularly
            if batch_idx % args.log_interval == 0 or batch_idx == len(test_loader) - 1:
                print(f'Test Epoch: {epoch} ', end='')
                print(f'[{batch_idx * images.shape[0]}/{len(test_loader.dataset)} ({100. * batch_idx / len(test_loader):.0f}%)]', end='')
                print(f'\tLoss: {loss.item()}')

    # Return the losses of this epoch
    return torch.tensor(all_losses).mean()
