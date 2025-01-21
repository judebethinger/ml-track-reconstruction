import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Model
from tensorflow.keras import models
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Reshape, Concatenate, Conv2DTranspose
from sklearn.metrics import precision_recall_fscore_support



num_events = 50
grid_size = (16,16)
max_tracks_per_event = 5
kernel_size_network1 = (3,3)
kernel_size_network2 = (3,3)



# Function for creating the 2D data grid per event
def prepare_data_grid(events, grid_size):
    X = [] 
    y = [] 
    
    for event in events:
        # Initialize to a grid with full 0
        grid = np.zeros(grid_size)
        
        for track in event['tracks']:
            for point in track:
                x, track_y = point 
                
                # Normalize coordinates to fit within the grid size
                grid_x = int(np.clip((x + 6) * (grid_size[0] / 12), 0, grid_size[0] - 1))
                grid_y = int(np.clip(track_y * (grid_size[1] / 6), 0, grid_size[1] - 1))
                
                # A 1 in the grid indicating a hit point
                grid[grid_y, grid_x] += 1

        # Put the noise hit points in the same grid
        for point in event['noise']:
            x, track_y = point
            grid_x = int(np.clip((x + 6) * (grid_size[0] / 12), 0, grid_size[0] - 1))
            grid_y = int(np.clip(track_y * (grid_size[1] / 6), 0, grid_size[1] - 1)) 
            
            # A 1 in the grid indicating a hit point
            grid[grid_y, grid_x] += 1 
        
        X.append(grid.flatten())
        
        # Number of true tracks as label 
        y.append(len(event['tracks']))

    X = np.array(X)
    y = np.array(y)
    
    return X, y



# Function for ccreating a data grid with labels
def prepare_data_grid_with_coordinates(events, grid_size):
    X = [] 
    y_hit_points = [] 
    y_track_membership = [] 
    
    for event in events:
        grid = np.zeros((grid_size[0], grid_size[1], 2))  # 2 channels: hit points and track membership
        
        for track_index, track in enumerate(event['tracks']):
            for point in track:
                x, track_y = point 
                
                # Normalize coordinates to fit within the grid size
                grid_x = int(np.clip((x + 6) * (grid_size[0] / 12), 0, grid_size[0] - 1))
                grid_y = int(np.clip(track_y * (grid_size[1] / 6), 0, grid_size[1] - 1))
                
                grid[grid_y, grid_x, 0] = 1  # Mark hit point in the first channel
                grid[grid_y, grid_x, 1] = track_index + 1  # Track membership in the second channel

        # Handle noise points
        for point in event['noise']:
            x, track_y = point
            grid_x = int(np.clip((x + 6) * (grid_size[0] / 12), 0, grid_size[0] - 1))
            grid_y = int(np.clip(track_y * (grid_size[1] / 6), 0, grid_size[1] - 1)) 
            
            grid[grid_y, grid_x, 0] = 1  # Mark hit point in the first channel
            grid[grid_y, grid_x, 1] = 0  # No track membership for noise in the second channel

        X.append(grid)

        # Create hit point labels
        hit_point_labels = np.zeros((grid_size[0], grid_size[1])) 
        for track_index, track in enumerate(event['tracks']):
            for point in track:
                x, track_y = point 
                grid_x = int(np.clip((x + 6) * (grid_size[0] / 12), 0, grid_size[0] - 1))
                grid_y = int(np.clip(track_y * (grid_size[1] / 6), 0, grid_size[1] - 1))
                
                hit_point_labels[grid_y, grid_x] = 1

        y_hit_points.append(hit_point_labels.flatten())
        y_track_membership.append(grid[:, :, 1].flatten().astype(int))
    
    return np.array(X), np.array(y_hit_points), np.array(y_track_membership)



# Function for generating random tracks
def generate_tracks(num_tracks, max_y=5):

    tracks = []
    
    for _ in range(num_tracks):
        start_x = random.uniform(-0.5, 0.5)
        start_y = 0
       
        angle = random.uniform(-np.pi / 4, np.pi / 4)  
        slope = np.tan(angle)
        
        track = [(start_x, start_y)]
        for y in range(1, max_y + 1):
            x = start_x + slope * y
            track.append((x, y))
        
        tracks.append(track)
    
    return tracks



# Function for generating random noise hit points
def generate_noise_points(num_points, max_x=6, max_y=6):
    noise_points = []
    for _ in range(num_points):
        x = random.uniform(-max_x, max_x)
        y = random.uniform(0, max_y)
        noise_points.append((x, y))
    return noise_points



# Function for creating the events
def generate_events(num_events, max_tracks_per_event, max_y=5): 
    events = []
    
    for _ in range(num_events):
        num_tracks = random.randint(0, max_tracks_per_event)
        tracks = generate_tracks(num_tracks, max_y)

        num_noise_points = random.randint(1, 3)
        noise_points = generate_noise_points(num_noise_points, max_x=6, max_y=max_y)

        event = {
            'tracks': tracks,
            'noise': noise_points
        }
        events.append(event)
    
    return events




# Function for the first neural network
def neural_network1(input_shape, max_tracks_per_event, kernel_size_network1):
    model = models.Sequential()
    model.add(Conv2D(32, (kernel_size_network1[0], kernel_size_network1[1]), activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (kernel_size_network1[0], kernel_size_network1[1]), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (kernel_size_network1[0], kernel_size_network1[1]), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(max_tracks_per_event+1, activation='softmax'))

    model.compile(optimizer=Adam(), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model



# Function for the second neural network
def neural_network2(input_shape, max_tracks_per_event, kernel_size_network2):
    grid_input = Input(shape=input_shape, name='grid_input')
    track_count_input = Input(shape=(1,), name='track_count_input')
    reshaped = Reshape((input_shape[0], input_shape[1], 2))(grid_input)
    
    x = Conv2D(16, (kernel_size_network2[0], kernel_size_network2[1]), activation='relu', padding='same')(reshaped)
    x = MaxPooling2D((2, 2))(x) 
    x = Conv2D(32, (kernel_size_network2[0], kernel_size_network2[1]), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (kernel_size_network2[0], kernel_size_network2[1]), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (kernel_size_network2[0], kernel_size_network2[1]), activation='relu', padding='same')(x)
    x = Conv2DTranspose(64, (kernel_size_network2[0], kernel_size_network2[1]), strides=(2, 2), padding='same')(x)
    x = Conv2D(32, (kernel_size_network2[0], kernel_size_network2[1]), activation='relu', padding='same')(x)
    x = Conv2DTranspose(32, (kernel_size_network2[0], kernel_size_network2[1]), strides=(2, 2), padding='same')(x)
    x = Conv2D(16, (kernel_size_network2[0], kernel_size_network2[1]), activation='relu', padding='same')(x)
    x = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(x)
    
    x_flattened = Flatten()(x)
    
    x_combined = Concatenate()([x_flattened, track_count_input])
    
    x_combined = Dense(128, activation='relu')(x_combined)
    
    track_membership_output = Conv2D(max_tracks_per_event+1, (1, 1), activation='softmax', padding='same', name='track_membership')(x)
    
    model = Model(inputs=[grid_input, track_count_input], outputs=track_membership_output)
    
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model
    


# Function for plotting the events, with true tracks, noise hit points, true number of tracks and predicted number of hit points
def plot_multiple_events(events, predictions, num_samples=1):
    plt.figure(figsize=(18, 8)) 
    plt.suptitle("Model 1 Sample Events", fontsize=16, fontweight='bold')

    
    for i in range(num_samples):
        event = events[i]
        true_tracks_count = len(event['tracks'])
        predicted_tracks_count = predictions[i]

        plt.subplot(1, num_samples, i + 1)
        plt.axhline(0, color='gray', lw=0.5, linestyle='--') 
        plt.axvline(0, color='gray', lw=0.5, linestyle='--')
        plt.grid(True, linestyle='--', alpha=0.6)

        for track in event['tracks']:
            x, y = zip(*track)
            plt.plot(x, y, marker='o', markersize=5, color='blue', linestyle='-', alpha=0.8, label='True Track')

        noise_x, noise_y = zip(*event['noise'])
        plt.scatter(noise_x, noise_y, color='red', s=50, label='Noise Point', alpha=0.7, edgecolor='black')

        plt.title(f'True: {true_tracks_count}\nPredicted: {predicted_tracks_count}', fontsize=12)
        plt.xlabel('X-axis', fontsize=10)
        plt.ylabel('Y-axis', fontsize=10)

        plt.xlim(-6, 6)
        plt.ylim(-1, 6)
        plt.gca().set_aspect('equal', adjustable='box')

        if i == 0:
            plt.legend(loc='upper left', fontsize=8, frameon=True, edgecolor='gray')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()



# Function for creating the confusion matrix
def ConfusionMatrix(y,y_pred, num_classes):
    conf_matrix = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(num_classes), yticklabels=np.arange(num_classes))
    plt.title('Confusion Matrix Model 1')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()



# Function for plotting the accuracy and loss graphs
def PlotAccuracyAndLoss(history):
    plt.figure(figsize=(14, 6))
    plt.suptitle("Model 1 Learning Curves", fontsize=16, fontweight='bold')

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o', linestyle='-')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o', linestyle='--')
    plt.title('Model 1 Accuracy', fontsize=14)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', marker='o', linestyle='-')
    plt.plot(history.history['val_loss'], label='Validation Loss', marker='o', linestyle='--')
    plt.title('Model 1 Loss', fontsize=14)
    plt.ylabel('Loss', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.95]) 
    plt.show()




# Function for plotting the tracks and their memberships
def plot_tracks_with_membership(true_tracks, predicted_outputs, noise_points, grid_size, num_samples=5):
    num_samples = min(num_samples, len(true_tracks))

    for i in range(num_samples):
        true_track = true_tracks[i]
        predicted_track_membership = predicted_outputs[i]
        noise = noise_points[i]

        predicted_track_membership_reshaped = predicted_track_membership.reshape(grid_size[0], grid_size[1], -1)
        predicted_track_indices = np.argmax(predicted_track_membership_reshaped, axis=-1)

        noise_y, noise_x = zip(*[
            (int(np.clip(point[1], 0, grid_size[1] - 1)),
             int(np.clip(point[0] + grid_size[0] // 2, 0, grid_size[0] - 1)))
            for point in noise
        ])

        # Create a new figure for each sample
        plt.figure(figsize=(20, 12))
        plt.suptitle(f"Model 2 Sample Event {i + 1}", fontsize=16, fontweight='bold')

        # Plot true tracks
        plt.subplot(2, 1, 1)
        plt.imshow(true_track.reshape(grid_size), cmap='Blues', alpha=0.8, interpolation='nearest')
        plt.scatter(noise_x, noise_y, color='orange', edgecolor='black', s=60, label='Noise Points', alpha=0.7)
        plt.title("True Track", fontsize=12)
        plt.axis('off') 

        # Plot predicted tracks
        plt.subplot(2, 1, 2)
        plt.imshow(predicted_track_indices, cmap='Reds', alpha=0.8, interpolation='nearest')
        plt.title("Predicted Track", fontsize=12)
        plt.axis('off')  

        plt.tight_layout(rect=[0, 0, 1, 0.95]) 
        plt.show()  # Show the plot for the current event



# Function for making the confusion matrix for neural network 2
def plot_confusion_matrix(true_labels, predicted_labels, num_classes):
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(num_classes), yticklabels=np.arange(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix Model 2')
    plt.show()



# Function for printing the classification report of neural network 2
def print_classification_report(true_labels, predicted_labels): 
    print("Track Membership Classification Report:")
    print(classification_report(true_labels, predicted_labels))



# Function for plotting the learning curves
def plot_learning_curves(history):
    plt.figure(figsize=(14, 6))
    plt.suptitle("Model 2 Learning Curves", fontsize=18, fontweight='bold', y=1.02)

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o', color='blue', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o', color='orange', linewidth=2)
    plt.title('Accuracy Over Epochs', fontsize=16, pad=15)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(visible=True, linestyle='--', linewidth=0.5, alpha=0.7)

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', marker='o', color='blue', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', marker='o', color='orange', linewidth=2)
    plt.title('Loss Over Epochs', fontsize=16, pad=15)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(visible=True, linestyle='--', linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    plt.show()



# Function for evaluating the precision and F1 score
def class_wise_metrics(true_labels, predicted_labels, num_classes):
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, labels=np.arange(num_classes))
    for i in range(num_classes):
        print(f"Class {i}: Precision: {precision[i]:.2f}, Recall: {recall[i]:.2f}, F1-Score: {f1[i]:.2f}")



# Function for plotting the performance of neural network 2
def evaluate_model_performance(true_track_membership, predicted_track_membership, num_classes, history):
    true_labels = true_track_membership.flatten()
    predicted_labels = predicted_track_membership.flatten()

    plot_learning_curves(history)

    plot_confusion_matrix(true_labels, predicted_labels, num_classes)

    print_classification_report(true_labels, predicted_labels)

    class_wise_metrics(true_labels, predicted_labels, num_classes)




# Function for testing new data on the second neural network
def test_new_data_second_neural_network(model2, grid_size, model1, max_tracks_per_event):
    num_new_events = 10
    new_events = generate_events(num_new_events, max_tracks_per_event)
    
    X_new, y_new, y_track_membership_new = prepare_data_grid_with_coordinates(new_events, grid_size)
    y_track_membership_new = np.array(y_track_membership_new).reshape(-1, grid_size[0], grid_size[1], 1)

    # Use the first network model to predict the number of tracks per event
    y_pred_probabilities = model1.predict(X_new)  
    y_pred_classes = np.argmax(y_pred_probabilities, axis=1) 

    num_tracks_per_event_new = y_pred_classes

    predicted_outputs_new = model2.predict([X_new, num_tracks_per_event_new])

    y_track_membership_new_reshaped = y_track_membership_new.reshape(-1, grid_size[0], grid_size[1])

    true_tracks_new = [y_track_membership_new_reshaped[i] for i in range(num_new_events)]
    noise_points_new = [new_events[i]['noise'] for i in range(num_new_events)]

    plot_tracks_with_membership(true_tracks_new, predicted_outputs_new, noise_points_new, grid_size)



# Function for the logic of the first neural network
def logic_first_network(events, grid_size, max_tracks_per_event, kernel_size_network1):
    X, y = prepare_data_grid(events, grid_size)

    # Reshape the data so it fits into the CNN
    X = X.reshape(X.shape[0], grid_size[0], grid_size[1], 1)

    input_shape = (grid_size[0], grid_size[1], 1)

    model = neural_network1((input_shape), max_tracks_per_event, kernel_size_network1)
    history = model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2)

    y_pred = np.argmax(model.predict(X), axis=1)

    print(classification_report(y, y_pred))

    model.summary()

    PlotAccuracyAndLoss(history)

    plot_multiple_events(events, y_pred, num_samples=5)

    for i in range(10):
        print(f"True: {y[i]}, Predicted: {y_pred[i]}")

    y_pred_probabilities = model.predict(X)
    y_pred_classes = np.argmax(y_pred_probabilities, axis=1)
    num_tracks_per_event = y_pred_classes

    ConfusionMatrix(y,y_pred, max_tracks_per_event+1)

    return num_tracks_per_event, model



# Function for the logic of the second neural network
def logic_second_network(events, grid_size, num_tracks_per_event, model1, max_tracks_per_event, kernel_size_network2):
    X, y, y_track_membership = prepare_data_grid_with_coordinates(events, grid_size)

    y_track_membership = np.array(y_track_membership).reshape(-1, grid_size[0], grid_size[1], 1)

    input_shape = (grid_size[0], grid_size[1], 2)

    model = neural_network2(input_shape, max_tracks_per_event, kernel_size_network2)

    history = model.fit([X, num_tracks_per_event], y_track_membership , epochs=20, batch_size=32, validation_split=0.2)

    num_samples_to_plot = 1
    sample_indices = np.random.choice(len(X), num_samples_to_plot, replace=False)
    predicted_outputs = model.predict([X[sample_indices], num_tracks_per_event[sample_indices]])

    y_track_membership_reshaped = y_track_membership.reshape(-1, grid_size[0], grid_size[1])

    true_tracks = [y_track_membership_reshaped[i] for i in sample_indices]
    noise_points_sample = [events[i]['noise'] for i in sample_indices] 

    plot_tracks_with_membership(true_tracks, predicted_outputs, noise_points_sample, grid_size)

    predicted_track_membership = np.argmax(predicted_outputs, axis=-1)

    true_track_membership = y_track_membership[sample_indices] 
        
    predicted_outputs_full = model.predict([X, num_tracks_per_event])

    predicted_track_membership_full = np.argmax(predicted_outputs_full, axis=-1)

    true_track_membership_full = y_track_membership.reshape(-1, grid_size[0], grid_size[1]) 

    true_labels_full = true_track_membership_full.flatten()
    predicted_labels_full = predicted_track_membership_full.flatten()

    mask = true_labels_full != 0 
    filtered_true_labels = true_labels_full[mask]
    filtered_predicted_labels = predicted_labels_full[mask]

    evaluate_model_performance(filtered_true_labels, filtered_predicted_labels, max_tracks_per_event+1, history)

    test_new_data_second_neural_network(model, grid_size, model1, max_tracks_per_event)
    


events = generate_events(num_events, max_tracks_per_event)
num_tracks_per_event, model1 = logic_first_network(events, grid_size, max_tracks_per_event, kernel_size_network1)
logic_second_network(events, grid_size, num_tracks_per_event, model1, max_tracks_per_event, kernel_size_network2)