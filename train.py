#!/usr/bin/env python
# coding: utf-8

# # Training of the number recognition model

# ## Importation des modules et définition des fonctions les plus importantes

# In[1]:


# Importation des modules 

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

# Définition des fonctions importantes
def reset_graph(seed=42):
    " Fonction de réinitialisation de graphe "
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
    
    
def shuffle_batch(X, y, batch_size):
    " Fonction qui permet de choisir melanger le dataset et retourner des mini-lots"
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch


# ### Importation du dataset et separation des données pour l'entrainement et l'évaluation

# In[2]:


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0

y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]


# ## Construction du graphe

# In[3]:


n_inputs = 28*28 # Nombres de pixels
n_hidden1 = 300
n_hidden2 = 50
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

# Mise en place des réseaux de neurones
with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1")
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name="hidden2")
    logits = tf.layers.dense(hidden2, n_outputs, name="logits")

# Définition de la fonction de perte avec l'entropie croisée
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")


# Entrainement du modèle
with tf.name_scope("train"):
    # Initialisation du taux d'apprentissage avec la planification par exponentielle
    initial_learning_rate = 0.1
    decay_steps = 10000
    decay_rate = 1/10
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(initial_learning_rate,global_step,
                                           decay_rate=decay_rate, decay_steps=decay_steps)
    
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
    training_op = optimizer.minimize(loss, global_step)



# Evaluation du moldèle 
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
    

# Fin de la phase de construction
init = tf.global_variables_initializer()
saver = tf.train.Saver()


# ## Exécution du graphe

# ### Définition des hyperparametres fixes

# In[4]:


n_epochs = 400
batch_size = 200


# ## Entrainement et vérification des résultats avec 5 exemples

# In[5]:


with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X:X_batch, y:y_batch})
        if epoch % 10 == 0:
            accuracy_val = accuracy.eval(feed_dict={X:X_batch, y:y_batch})
            print(epoch, "Validation accuracy : ", accuracy_val)
        
    save_path = saver.save(sess, "./models/my_model_final.ckpt")
    
    # Utilisation de 5 images pour le test
    index = np.random.randint(0, 4995)
    images= X_train[index: index+5]
    
    predictions = tf.argmax(logits, 1)
    predicted_labels = predictions.eval(feed_dict={X:images})
    
    for i in range(len(images)):
        plt.bone()
        plt.imshow(images[i].reshape([28, 28]))
        plt.show()
        
        print("La valeur prédite est ", predicted_labels[i])
                      


# In[ ]:




