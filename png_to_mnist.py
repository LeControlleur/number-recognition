#!/usr/bin/env python
# coding: utf-8

# # Module qui permet de convertir une image png en MNIST

# ## Importation des librairies

# In[2]:


from PIL import Image, ImageFilter


# ## Définition de la fonction principale de conversion

# In[5]:


def imagePrepare(path):
    """ 
    Cette fonction permet de convertir une image en format MNIST 28x28.
    On envoie en parametre l'adesse de l'image
    """
    im = Image.open(path).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255)) # Création d'un canevas blanc de 28*28pixels
    
    if width > height:
        # La largeur, plus gande, deviendra 20 px
        nheight = int(round((20.0 / width * height), 0)) # Rédimensionnement de la hauteur suivant un ratio
        
        if (nheight == 0):
            nheight = 1
            
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0)) # Calcul de la position horizontale
        newImage.paste(img, (4, wtop)) # Ajout de l'image rédimensionnée dans le canevas
        
    else :
        # La hauteur est plus grnade et deviens donc 20 px
        # Le reste du code est le meme que dans le cas précédent
        nwidth = int(round((20.0 / height * width), 0)) # Rédimensionnement de la hauteur suivant un ratio
        
        if (nwidth == 0):
            nwidth = 1
            
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0)) # Calcul de la position horizontale
        newImage.paste(img, (4, wleft)) # Ajout de l'image rédimensionnée dans le canevas
    
    # Sauvegarde de la nouvelle image pour un test
    newImage.save("sample.png")
    
    tv = list(newImage.getdata()) # Collecte des valeurs des pixels
    
    # Normalisation des pixels à 0(blanc) ou 1 (noir)
    tva = [(255 - x) * 1.0 / 255 for x in tv]
    
    return tva

