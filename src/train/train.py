import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import torch
import wandb


def train_epoch_Centerspeed(training_loader, net, optimizer, loss_fn, device = 'cpu', use_wandb=False, pdf=None):
    running_loss = 0.
    last_loss = 0.
    print("Using Train epoch Centerspeed")

    for i, data in enumerate(training_loader):
        
        inputs, gts, data, is_free = data
        inputs = inputs.to(device)
        gts = gts.to(device)
        data = data.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        output_hms, output_data = net(inputs) 
        gts = gts.unsqueeze(1)
        # Compute the loss and its gradients
        loss = loss_fn(output_hms,output_data, gts, data, is_free)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        
        last_loss = loss.item() # loss per batch
        print('--batch {} loss: {}'.format(i + 1, last_loss))
        if use_wandb:
            wandb.log({"batch_loss": last_loss/len(inputs)})#log the average loss per batch
        
        if pdf is not None: #Can be used to save the outputs of the model to a pdf, one plot per batch
            plt.imshow(output_hms[0].squeeze(0).detach().numpy(), origin='lower')
            plt.title( f'Batch: {i+1}, Loss: {last_loss}')
            pdf.savefig()
            plt.close()
    return last_loss

###################### Old functions ##################################

def train_epoch(training_loader, net, optimizer, loss_fn, use_wandb=False):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, gts,_ = data
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = net(inputs) 
        gts = gts.unsqueeze(1)#add a channel dimension to fit the loss function
        # Compute the loss and its gradients
        loss = loss_fn(outputs, gts)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        
        last_loss = running_loss # loss per batch
        print('  batch {} loss: {}'.format(i + 1, last_loss))
        if use_wandb:
            wandb.log({"batch_loss": last_loss/len(inputs)})#log the average loss per batch
        running_loss = 0.

    return last_loss

def train_epoch_Centerspeed(training_loader, net, optimizer, loss_fn, device = 'cpu', use_wandb=False, pdf=None):
    running_loss = 0.
    last_loss = 0.
    print("Using Train epoch Centerspeed")

    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        
        inputs, gts, data, is_free = data
        inputs = inputs.to(device)
        gts = gts.to(device)
        data = data.to(device)
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        output_hms, output_data = net(inputs) 
        gts = gts.unsqueeze(1)
        # Compute the loss and its gradients
        loss = loss_fn(output_hms,output_data, gts, data, is_free)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        
        last_loss = loss.item() # loss per batch
        print('  batch {} loss: {}'.format(i + 1, last_loss))
        if use_wandb:
            wandb.log({"batch_loss": last_loss/len(inputs)})#log the average loss per batch
        if pdf is not None: 
            plt.imshow(output_hms[0].squeeze(0).detach().numpy(), origin='lower')
            plt.title( f'Batch: {i+1}, Loss: {last_loss}')

            pdf.savefig()
            plt.close()
    return last_loss

def train_epoch_cs2(pdf, training_loader, net, optimizer, loss_fn, use_wandb=False):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, gts, data = data
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        output_hms, output_data = net(inputs) 
        gts = gts.unsqueeze(1)
        # Compute the loss and its gradients
        loss = loss_fn(output_hms,output_data, gts, data)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        
        last_loss = loss.item() # loss per batch
        print('  batch {} loss: {}'.format(i + 1, last_loss))
        if use_wandb:
            wandb.log({"batch_loss": last_loss})#log the average loss per batch
        plt.imshow(output_hms[0].squeeze(0).detach().numpy(), origin='lower')
        plt.title( f'Batch: {i+1}, Loss: {last_loss}')

        pdf.savefig()
        plt.close()
    return last_loss

def train_epoch_hm(pdf, training_loader, net, optimizer, loss_fn, use_wandb=False):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, gts, data = data
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        output_hms, output_data = net(inputs) 
        gts = gts.unsqueeze(1)
        # Compute the loss and its gradients
        loss = loss_fn(output_hms,output_data, gts, data)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        
        last_loss = loss.item() # loss per batch
        print('  batch {} loss: {}'.format(i + 1, last_loss))
        if use_wandb:
            wandb.log({"batch_loss": last_loss})#log the average loss per batch
        plt.imshow(output_hms[0].squeeze(0).detach().numpy(), origin='lower')
        plt.title( f'Batch: {i+1}, Loss: {last_loss}')

        pdf.savefig()
        plt.close()
    return last_loss

def train_epoch_head(pdf, training_loader, net_hm, net_head, optimizer, loss_fn, use_wandb=False):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, gts, data = data
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        output_hms, feature_map = net_hm(inputs) 

        head_output = net_head(feature_map)
        # Compute the loss and its gradients
        loss = loss_fn(output_hms,head_output, gts, data[:,2:])
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        
        last_loss = loss.item() # loss per batch
        print('  batch {} loss: {}'.format(i + 1, last_loss))
        if use_wandb:
            wandb.log({"batch_loss": last_loss})#log the average loss per batch
        plt.imshow(output_hms[0].squeeze(0).detach().numpy(), origin='lower')
        plt.title( f'Batch: {i+1}, Loss: {last_loss}')

        pdf.savefig()
        plt.close()
    return last_loss

def train_hm(net, training_loader, validation_loader, optimizer, loss_fn, epochs, use_wandb=False):
    if use_wandb:
        assert wandb.run is not None, "No wandb run found!"

    # Initializing in a separate cell so we can easily add more epochs to the same run
    epoch_number = 0

    EPOCHS = epochs

    best_vloss = 1_000_000.
    pdf_pages = PdfPages('centerspeed.pdf')

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        net.train(True)
        avg_loss = train_epoch_hm(pdf_pages, training_loader, net, optimizer, loss_fn, use_wandb)


        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        net.eval()

        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                inputs, gts, data = vdata
                # Zero your gradients for every batch!
                optimizer.zero_grad()

                # Make predictions for this batch
                output_hms, output_data = net(inputs) 
                gts = gts.unsqueeze(1)
                # Compute the loss and its gradients
                loss = loss_fn(output_hms,output_data, gts, data[:,2:])
                running_vloss += loss.item()
        avg_vloss = running_vloss / (i + 1)
        if use_wandb:
            wandb.log({"train-loss": avg_loss, "validation-loss": avg_vloss})
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        epoch_number += 1

    pdf_pages.close()

def train_head(net_hm, net_head, training_loader, validation_loader, optimizer, loss_fn, epochs, use_wandb=False):
    if use_wandb:
        assert wandb.run is not None, "No wandb run found!"

    # Initializing in a separate cell so we can easily add more epochs to the same run
    epoch_number = 0

    EPOCHS = epochs

    best_vloss = 1_000_000.
    pdf_pages = PdfPages('centerspeed.pdf')

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        net_head.train(True)
        net_hm.train(False)
        avg_loss = train_epoch_head(pdf_pages, training_loader, net_hm, net_head, optimizer, loss_fn, use_wandb)


        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        net_head.eval()

        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                inputs, gts, data = vdata
                # Zero your gradients for every batch!
                optimizer.zero_grad()

                # Make predictions for this batch
                output_hms, feature_map = net_hm(inputs) 
                head_output = net_head(feature_map)
                # Compute the loss and its gradients
                loss = loss_fn(output_hms,head_output, gts, data[:,2:])
                running_vloss += loss.item()
        avg_vloss = running_vloss / (i + 1)
        if use_wandb:
            wandb.log({"train-loss": avg_loss, "validation-loss": avg_vloss})
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        epoch_number += 1

    pdf_pages.close()

def train_epoch_Centerspeed_decay(pdf, training_loader, net, optimizer, loss_fn, use_wandb=False, fac_hm=1, fac_data=1):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, gts, data = data
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        output_hms, output_data = net(inputs) 
        gts = gts.unsqueeze(1)
        # Compute the loss and its gradients
        loss = loss_fn(output_hms,output_data, gts, data[:,2:], fac_hm, fac_data)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        
        last_loss = loss.item() # loss per batch
        print('  batch {} loss: {}'.format(i + 1, last_loss/len(inputs)))
        if use_wandb:
            wandb.log({"batch_loss": last_loss/len(inputs)})#log the average loss per batch
        plt.imshow(output_hms[0].squeeze(0).detach().numpy(), origin='lower')
        plt.title( f'Batch: {i+1}, Loss: {last_loss}')

        pdf.savefig()
        plt.close()

        #Update the loss factors
        fac_hm *= 0.99
    return last_loss

def train_CenterSpeed_Decay(net, training_loader, validation_loader, optimizer, loss_fn, epochs, use_wandb=False):
    if use_wandb:
        assert wandb.run is not None, "No wandb run found!"

    # Initializing in a separate cell so we can easily add more epochs to the same run
    epoch_number = 0

    EPOCHS = epochs

    best_vloss = 1_000_000.
    pdf_pages = PdfPages('centerspeed.pdf')

    weight_hm = 1
    weight_data = 1

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        net.train(True)
        avg_loss = train_epoch_Centerspeed_decay(pdf_pages, training_loader, net, optimizer, loss_fn, use_wandb, fac_hm=weight_hm, fac_data=weight_data)


        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        net.eval()

        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                inputs, gts, data = vdata
                # Zero your gradients for every batch!
                optimizer.zero_grad()

                # Make predictions for this batch
                output_hms, output_data = net(inputs) 
                gts = gts.unsqueeze(1)
                # Compute the loss and its gradients
                loss = loss_fn(output_hms,output_data, gts, data[:,2:], weight_hm, weight_data)
                running_vloss += loss.item()
        avg_vloss = running_vloss / (i + 1)
        if use_wandb:
            wandb.log({"train-loss": avg_loss, "validation-loss": avg_vloss})
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        torch.save(net.state_dict(), 'centerspeed_epoch_' + str(epoch) + '.pth')

        epoch_number += 1

    pdf_pages.close()
