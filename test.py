#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


# ## Importation du modèle déjà entrainé

# In[2]:


saver = tf.train.import_meta_graph("./models/my_model_final.ckpt.meta")


# In[3]:


for op in tf.get_default_graph().get_operations():
    print(op.name)



# ## Utilisation du modèle

# ### Importation des variables à utiliser

# In[45]:


X = tf.get_default_graph().get_tensor_by_name("X:0")

logits = tf.get_default_graph().get_tensor_by_name("dnn/logits/BiasAdd:0")
predictions = tf.argmax(logits, 1)


# ## Importation de la librairie et tranformation des fichiers png en mnist, puis évaluation de l'image

# In[46]:


import png_to_mnist


# In[56]:


init = tf.global_variables_initializer()

sess = tf.InteractiveSession()
    
init.run()

choix = "O"
while choix != "n":
    image = input("Veuillez entrer le lien de l'image .png :" )

    image = np.array(png_to_mnist.imagePrepare(image))

    img = np.array([image])
    predicted_labels = predictions.eval(feed_dict={X: img})
    print("La valeur prédite est : ", predictions.eval(feed_dict={X: img}))

    plt.gray()
    plt.imshow(image.reshape([28, 28]))
    plt.show()

    
    
    choix = input("Voulez-vous faire un autre test ? (O/n) ")
    print("\n\n\n")
    


# In[ ]:




