import tensorflow as tf
import keras
from keras import backend as K
from keras import Sequential
from keras.src.layers import Dense, RandomRotation
from keras.src.losses import CategoricalCrossentropy
from keras.src.metrics import CategoricalAccuracy, Mean, SparseCategoricalAccuracy
from model.ResNet import ResNet

class ContrastiveModel(keras.Model):
    def __init__(self, num_classes, width=512, temperature=0.1):
        super().__init__()
        self.random_rotate = RandomRotation(0.2)
        self.temperature = temperature
        self.encoder = ResNet()
        # Non-linear MLP as projection head
        self.projection_head = Sequential(
            [
                keras.Input(shape=(width,)),
                Dense(width, activation="relu"),
                Dense(width),
            ],
            name="projection_head",
        )
        # Single dense layer for linear probing
        self.linear_probe = Sequential(
            [keras.Input(shape=(width,)), Dense(num_classes, activation='softmax')],
            name="linear_probe",
        )


    def augment_image(self, image, augment_type='color-jitter'):
        if augment_type == 'color-jitter':
            image = tf.image.random_brightness(image, max_delta=0.5)
            image = tf.image.random_contrast(image, lower=0.1, upper=0.2)
        elif augment_type == 'random-rotate':
            image = self.random_rotate(image)
        return image
    
    def compile(self, contrastive_optimizer, probe_optimizer, **kwargs):
        super().compile(**kwargs)

        self.contrastive_optimizer = contrastive_optimizer
        self.probe_optimizer = probe_optimizer

        # self.contrastive_loss will be defined as a method
        self.probe_loss = CategoricalCrossentropy()

        self.contrastive_loss_tracker = Mean(name="c_loss")
        self.contrastive_accuracy = SparseCategoricalAccuracy(
            name="c_acc"
        )
        self.probe_loss_tracker = Mean(name="p_loss")
        self.probe_accuracy = CategoricalAccuracy(name="p_acc")

    @property
    def metrics(self):
        return [
            self.contrastive_loss_tracker,
            self.contrastive_accuracy,
            self.probe_loss_tracker,
            self.probe_accuracy,
        ]

    def contrastive_loss(self, projections_1, projections_2):
        # Cosine similarity: the dot product of the l2-normalized feature vectors
        projections_1 = K.normalize(projections_1, axis=1)
        projections_2 = K.normalize(projections_2, axis=1)
        similarities = (
            K.matmul(projections_1, K.transpose(projections_2)) / self.temperature
        )

        # The similarity between the representations of two augmented views of the
        # same image should be higher than their similarity with other views
        batch_size = K.shape(projections_1)[0]
        contrastive_labels = K.arange(batch_size)
        self.contrastive_accuracy.update_state(contrastive_labels, similarities)
        self.contrastive_accuracy.update_state(
            contrastive_labels, K.transpose(similarities)
        )

        # The temperature-scaled similarities are used as logits for cross-entropy
        # a symmetrized version of the loss is used here
        loss_1_2 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, similarities, from_logits=True
        )
        loss_2_1 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, K.transpose(similarities), from_logits=True
        )
        return (loss_1_2 + loss_2_1) / 2

    def train_step(self, data):
        images, labels = data

        # Each image is augmented twice, differently
        augmented_images_1 = self.augment_image(images, augment_type='random-rotate')
        augmented_images_2 = self.augment_image(images, augment_type='color-jitter')
        with tf.GradientTape() as tape:
            # Encoder -> MLP -> Contrastive loss
            features_1 = self.encoder(augmented_images_1, training=True)
            features_2 = self.encoder(augmented_images_2, training=True)

            projections_1 = self.projection_head(features_1, training=True)
            projections_2 = self.projection_head(features_2, training=True)
            contrastive_loss = self.contrastive_loss(projections_1, projections_2)
        
        gradients = tape.gradient(
            contrastive_loss,
            self.encoder.trainable_weights + self.projection_head.trainable_weights,
        )
        self.contrastive_optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights + self.projection_head.trainable_weights,
            )
        )
        self.contrastive_loss_tracker.update_state(contrastive_loss)

        with tf.GradientTape() as tape:
            # the encoder is used in inference mode here to avoid regularization
            # and updating the batch normalization paramers if they are used
            features = self.encoder(images, training=False)
            class_logits = self.linear_probe(features, training=True)
            probe_loss = self.probe_loss(labels, class_logits)
        gradients = tape.gradient(probe_loss, self.linear_probe.trainable_weights)
        self.probe_optimizer.apply_gradients(
            zip(gradients, self.linear_probe.trainable_weights)
        )
        self.probe_loss_tracker.update_state(probe_loss)
        self.probe_accuracy.update_state(labels, class_logits)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        labeled_images, labels = data

        features = self.encoder(labeled_images, training=False)
        class_logits = self.linear_probe(features, training=False)
        probe_loss = self.probe_loss(labels, class_logits)
        self.probe_loss_tracker.update_state(probe_loss)
        self.probe_accuracy.update_state(labels, class_logits)

        # Only the probe metrics are logged at test time
        return {m.name: m.result() for m in self.metrics[2:]}