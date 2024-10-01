#!/usr/bin/env python
# coding: utf-8

# In[44]:


import panda as pd
df = pd.read_csv("data_isep.csv")
df.set_index("Mois",drop=True)
df


# In[45]:


#question 1
df_no_month = df.drop(columns = "Mois")

df_summed = df_no_month.sum()
total = df_summed.sum()
nb_col = len(df_no_month.columns)
print("VOILA LA LISTE DES OBSERVATIONS : ")
print()
print(df_summed)


print()
print("Il y a " + str(nb_col) + " espèces différentes qui ont été obsérvées")
print("Il y a un total de " + str(total) + " individus qui ont été obsérvés")
print("L'espèce la plus obsérvée à été : " + df_summed.idxmax()+" avec "+str(df_summed.max())+" observations.")
print("L'espèce la moins obsérvée à été : " + df_summed.idxmin()+" avec "+str(df_summed.min())+" observations.")



    
    


# In[46]:


#Question 2

zero_count_species = (df_no_month == 0).sum()
num_zero_species = (zero_count_species >0).value_counts()[0]
print("Il y a " + str(num_zero_species) +  " espèces qui ont passé au moins un mois sans être observées")
print("Ces espèces ont été supprimés de la base de données")

zero_cont_months = (df_no_month == 0).any(axis=1)
num_zero_months = zero_cont_months.sum()
print("Il y a " + str(num_zero_months) +  " mois ou au moins une éspèce n'a pas été observée")



zero_cols = (df_no_month == 0).any()

df_no_month = df_no_month.drop(zero_cols[zero_cols == True].index, axis=1)
df_no_month


# In[47]:


#Question 3
print("Voici la variance pour chaque espèces : ")
print()
print(df_no_month.var())

print()


print("L'espèce avec la plus grande variance est : " + df_no_month.var().idxmin() + " avec une variance de " + str(df_no_month.var().min()) )
print("L'espèce avec la plus petite variance est : " + df_no_month.var().idxmax() + " avec une variance de " + str(df_no_month.var().max()) )


# In[110]:


#question 4 
import matplotlib.pyplot as plt
df_per = df[["Mois","Perruche à collier"]]
mois = df["Mois"]

plt.bar(mois, df["Perruche à collier"],color = 'orange', edgecolor = 'black', linewidth = 2)
plt.xticks(rotation=90)
plt.show()

print("Voici les observations des perruches à  collier : ")
print(df_per.to_string(index=False))
print()
mean_perr = df_per["Perruche à collier"].mean()
median_perr = df_per["Perruche à collier"].median()
print("La moyenne de perruche observées par mois est de : " + str(mean_perr))
print("La mediane des perruche observées par mois est de : " + str(median_perr))
print()
obs_max = df_per.max()
obs_min = df_per.min()
print("Le mois ou le plus de perruche ont été observées a été "+obs_max["Mois"]+" avec "+str(obs_max["Perruche à collier"])+" perruches" )
print("Le mois ou le moins de perruche ont été observées a été "+obs_min["Mois"]+" avec "+str(obs_min["Perruche à collier"])+" perruches" )
ecart_type = df_per["Perruche à collier"].std()
print("L'écart type de ces observations est de "+str(ecart_type))


#On peut voir que la moyenne et la mediane sont très similaire, les valeurs oscille donc gobalement autour de 8," +"\n" +
#mais on peut voir que l'ecart type est relativement elevé, donc elles oscille autour de 8 mais avec de grand ecart "


# In[131]:


#question 5

import matplotlib.pyplot as plt

plt.figure()
df_no_perr=df.drop(columns="Perruche à collier")
df_no_perr.set_index("Mois",drop=True)
df_no_perr_mean = df_no_perr.drop(columns="Mois").mean(axis=1)

df[["Perruche à collier","Mois"]].plot(color='r',linewidth=5.0)
df_no_perr_mean.plot(x="Mois")

plt.xticks(rotation=90)
plt.legend(bbox_to_anchor=(1.0, 1.0))


# In[137]:


#question 6
correl = df_no_month.corr()["Perruche à collier"].drop("Perruche à collier")
print("Voici les coefficients de corrélation entre  les observations des Perruches à collier et les autres espèces :")
print()
print(correl)
print()
esp_cor_max = correl.idxmax()
cor_max = correl.max()
esp_cor_min = correl.idxmin()
cor_min = correl.min()
print("L'espèce avec le plus haut coefficient est "+esp_cor_max+" avec un coefficient de "+str(cor_max))
print("L'espèce avec le plus bas coefficient est "+esp_cor_min+" avec un coefficient de "+str(cor_min))


# Créer une figure et un axe
fig, ax = plt.subplots()

# Tracer les données de chaque espèce d'oiseau
df[["Perruche à collier", "Mois"]].plot(color='black', linewidth=5, x="Mois", ax=ax)
df[esp_cor_max].plot(color='r', linewidth=3, ax=ax)
df[esp_cor_min].plot(color='blue', linewidth=3, ax=ax)

# Obtenir les mois uniques
mois_uniques = df["Mois"].unique()

# Définir une liste de couleurs pour les lignes verticales
couleurs = ['green', 'orange', 'purple', 'cyan', 'red', 'blue', 'yellow', 'magenta', 'teal', 'pink', 'lime', 'indigo']
  # Ajoutez autant de couleurs que nécessaire

# Tracer une ligne verticale pour chaque mois avec une couleur différente
for i, mois in enumerate(mois_uniques):
    ax.axvline(x=mois, color=couleurs[i % len(couleurs)], linestyle='--')

# Légende
plt.legend(bbox_to_anchor=(1.0, 1.0))

# Afficher le graphe
plt.show()


# In[51]:


#question 7
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Circle

figure, axes = plt.subplots()

f = df_no_month.columns


f = df.loc[:, f].values

pca = PCA(n_components=2)

f_sc = StandardScaler().fit_transform(f)

pc = pca.fit_transform(f_sc)

explanied_v = pca.explained_variance_ratio_


pca_df = pd.DataFrame(data = pc, columns = ['pc_1', 'pc_2'])

print(pca_df)
print(pca_df['pc_1'].var())
print(pca_df['pc_2'].var())


    
for i in range(12): 
    plt.scatter(pca_df['pc_1'][i],pca_df['pc_2'][i])

plt.legend(["janvier","fevrier","mars","avril","mai","juin",
            "juillet","aout","septembre","octobre","novembre","decembre"],bbox_to_anchor=(1.02, 1.05))
plt.xlabel('Comp prin 1, var ex : '+str(explanied_v[0]))
plt.ylabel('Comp prin 2, var ex : '+str(explanied_v[1]))

circle = Circle((0,0),1,fill=False)
axes.add_artist(circle)


# In[143]:


#question 11
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error

X = df["Perruche à collier"].values.reshape(-1, 1)
y = df[esp_cor_max].values
regression = LinearRegression()
regression.fit(X, y)
beta_0 = regression.intercept_
beta_1 = regression.coef_[0]


plt.scatter(X, y, color='b', label='Données observées')


x_values = X
y_values = beta_0 + beta_1 * x_values

plt.plot(x_values, y_values, color='r', label='Régression linéaire')

plt.xlabel(esp_cor_max)
plt.ylabel('Perruches à collier')
plt.legend()

print("β0 = " + str(beta_0))
print("β1 = " + str(beta_1))    

precision = regression.score(X,y)
print("Precision : ",precision)

# Coefficient R² non ajusté
y_pred = regression.predict(X)
r2 = r2_score(y, y_pred)
print("R² non ajusté :", r2)

# Coefficient R² ajusté
n = len(y)
p = 1  # Nombre de variables explicatives (dans ce cas, 1)
r2_adjusted = 1 - (1 - r2) * (n - 1) / (n - p - 1)
print("R² ajusté :", r2_adjusted)


# Prédiction
y_pred = regression.predict(X)

# Erreur quadratique moyenne (RMSE)
rmse = np.sqrt(mean_squared_error(y, y_pred))
print("RMSE :", rmse)


# In[145]:


#question 12
import numpy as np
import scipy.stats as stats
# Obtenez les prédictions du modèle
y_pred = regression.predict(X)

# Calcul des résidus
residuals = y - y_pred

# Calcul de la variance résiduelle
residual_variance = np.sum(residuals**2) / (len(X) - 2)

# Calcul de l'écart-type des coefficients
beta_0_std = np.sqrt(residual_variance) * np.sqrt((1 / len(X)) + (np.mean(X)**2 / np.sum((X - np.mean(X))**2)))
beta_1_std = np.sqrt(residual_variance) / np.sqrt(np.sum((X - np.mean(X))**2))

# Calcul de l'intervalle de confiance à 90% pour beta_0
t_value = stats.t.ppf(0.90, len(X) - 2)
beta_0_interval = [beta_0 - t_value * beta_0_std, beta_0 + t_value * beta_0_std]

# Calcul de l'intervalle de confiance à 90% pour beta_1
beta_1_interval = [beta_1 - t_value * beta_1_std, beta_1 + t_value * beta_1_std]

print("Intervalle de confiance à 90% pour beta_0 :", beta_0_interval)
print("Intervalle de confiance à 90% pour beta_1 :", beta_1_interval)


# In[69]:


#question 13


# Créez une instance du modèle de régression linéaire
regression = LinearRegression()

# Ajustez le modèle aux données
regression.fit(X, y)

# Calcul des prédictions du modèle
y_pred = regression.predict(X)

# Calcul des résidus
residuals = y - y_pred

# Calcul de la variance résiduelle
residual_variance = np.sum(residuals**2) / (len(X) - 2)

# Calcul de l'écart-type des coefficients
beta_1_std = np.sqrt(residual_variance) / np.sqrt(np.sum((X - np.mean(X))**2))

# Calcul de la statistique de test (t-value)
t_value = beta_1 / beta_1_std

# Calcul de la valeur p
p_value = 2 * (1 - stats.t.cdf(np.abs(t_value), len(X) - 2))

# Comparaison avec la valeur critique (niveau de confiance de 5%)
alpha = 0.05
if p_value < alpha:
    print("Le coefficient beta_1 est significativement non nul. Il y a une relation linéaire significative entre les deux variables.")
else:
    print("Le coefficient beta_1 n'est pas significativement non nul. Il n'y a pas de relation linéaire significative entre les deux variables.")


# In[70]:


#questipon 14


X = df["Perruche à collier"].values.reshape(-1, 1)
y = df[esp_cor_min].values
regression = LinearRegression()
regression.fit(X, y)
beta_0 = regression.intercept_
beta_1 = regression.coef_[0]


plt.scatter(X, y, color='b', label='Données observées')


x_values = X
y_values = beta_0 + beta_1 * x_values

plt.plot(x_values, y_values, color='r', label='Régression linéaire')

plt.xlabel(esp_cor_min)
plt.ylabel('Perruches à collier')
plt.legend()

print("β0 = " + str(beta_0))
print("β1 = " + str(beta_1))    


R2_non_ajusté = regression.score(X, y)

# Calcul du coefficient de détermination R2 ajusté
n = len(X)
p = 1  # Nombre de variables indépendantes (dans cet exemple, nous en avons une seule)
R2_ajusté = 1 - (1 - R2_non_ajusté) * (n - 1) / (n - p - 1)

print("R2 non ajusté :", R2_non_ajusté)
print("R2 ajusté :", R2_ajusté)



y_pred = regression.predict(X)

# Calcul des résidus
residuals = y - y_pred

# Calcul de la variance résiduelle
residual_variance = np.sum(residuals**2) / (len(X) - 2)

# Calcul de l'écart-type des coefficients
beta_0_std = np.sqrt(residual_variance) * np.sqrt((1 / len(X)) + (np.mean(X)**2 / np.sum((X - np.mean(X))**2)))
beta_1_std = np.sqrt(residual_variance) / np.sqrt(np.sum((X - np.mean(X))**2))

# Calcul de l'intervalle de confiance à 90% pour beta_0
t_value = stats.t.ppf(0.95, len(X) - 2)
beta_0_interval = [beta_0 - t_value * beta_0_std, beta_0 + t_value * beta_0_std]

# Calcul de l'intervalle de confiance à 90% pour beta_1
beta_1_interval = [beta_1 - t_value * beta_1_std, beta_1 + t_value * beta_1_std]

print("Intervalle de confiance à 90% pour beta_0 :", beta_0_interval)
print("Intervalle de confiance à 90% pour beta_1 :", beta_1_interval)



# Créez une instance du modèle de régression linéaire
regression = LinearRegression()

# Ajustez le modèle aux données
regression.fit(X, y)

# Calcul des prédictions du modèle
y_pred = regression.predict(X)

# Calcul des résidus
residuals = y - y_pred

# Calcul de la variance résiduelle
residual_variance = np.sum(residuals**2) / (len(X) - 2)

# Calcul de l'écart-type des coefficients
beta_1_std = np.sqrt(residual_variance) / np.sqrt(np.sum((X - np.mean(X))**2))

# Calcul de la statistique de test (t-value)
t_value = beta_1 / beta_1_std

# Calcul de la valeur p
p_value = 2 * (1 - stats.t.cdf(np.abs(t_value), len(X) - 2))

# Comparaison avec la valeur critique (niveau de confiance de 5%)
alpha = 0.05
if p_value < alpha:
    print("Le coefficient beta_1 est significativement non nul. Il y a une relation linéaire significative entre les deux variables.")
else:
    print("Le coefficient beta_1 n'est pas significativement non nul. Il n'y a pas de relation linéaire significative entre les deux variables.")


# In[98]:


#question 15
df_no_month_no_perr = df_no_month.drop(columns="Perruche à collier")
R2_ajust_max = 0
bird_max = ""
R2_ajust_min = 9
bird_min = ""

for bird in df_no_month_no_perr.columns:

    X = df["Perruche à collier"].values.reshape(-1, 1)
    
    y = df_no_month[bird].values
    regression = LinearRegression()
    regression.fit(X, y)
    beta_0 = regression.intercept_
    beta_1 = regression.coef_[0]


    plt.scatter(X, y, label=bird)


    x_values = X
    y_values = beta_0 + beta_1 * x_values

    plt.plot(x_values, y_values, label=bird)

    plt.xlabel("autres oiseaux")
    plt.ylabel('Perruches à collier')
    plt.legend()
    
    
    R2_non_ajust = regression.score(X, y)


    n = len(X)
    p = 1  
    R2_ajust = 1 - (1 - R2_non_ajust) * (n - 1) / (n - p - 1)

    if (R2_ajust>R2_ajust_max):
        R2_ajust_max = R2_ajust
        bird_max = bird
    if (R2_ajust<R2_ajust_min):
        R2_ajust_min = R2_ajust
        bird_min = bird    
    
    
    
plt.legend(bbox_to_anchor=(1.0, 1.0))


print("R2 ajusté max  :", R2_ajust_max)
print("Pour l'oiseau :",bird_max)
print("R2 ajusté min  :", R2_ajust_min)
print("Pour l'oiseau :",bird_min)


# In[112]:


#question 16



#question 15
df_no_month_no_perr = df_no_month.drop(columns="Perruche à collier")

good_bird=[]

#determiner lesquels ont une regression linéaire
for bird in df_no_month_no_perr.columns:

    X = df["Perruche à collier"].values.reshape(-1, 1)
    
    y = df_no_month[bird].values
    regression = LinearRegression()
    regression.fit(X, y)
    beta_0 = regression.intercept_
    beta_1 = regression.coef_[0]


    
    
    # Calcul des prédictions du modèle
    y_pred = regression.predict(X)

    # Calcul des résidus
    residuals = y - y_pred

    # Calcul de la variance résiduelle
    residual_variance = np.sum(residuals**2) / (len(X) - 2)

    # Calcul de l'écart-type des coefficients
    beta_1_std = np.sqrt(residual_variance) / np.sqrt(np.sum((X - np.mean(X))**2))

    # Calcul de la statistique de test (t-value)
    t_value = beta_1 / beta_1_std

    # Calcul de la valeur p
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_value), len(X) - 2))

    # Comparaison avec la valeur critique (niveau de confiance de 5%)
    alpha = 0.05
    if p_value < alpha:
        good_bird.append(bird)
    else:
        pass
        

        
        
        
        
#affichage des bon oiseaux
for bird in good_bird:

    X = df["Perruche à collier"].values.reshape(-1, 1)
    
    y = df_no_month[bird].values
    regression = LinearRegression()
    regression.fit(X, y)
    beta_0 = regression.intercept_
    beta_1 = regression.coef_[0]


    plt.scatter(X, y, label=bird)


    x_values = X
    y_values = beta_0 + beta_1 * x_values

    plt.plot(x_values, y_values, label=bird)

    plt.xlabel("autres oiseaux")
    plt.ylabel('Perruches à collier')
    plt.legend()
    

    
    
       
    
    
plt.legend(bbox_to_anchor=(1.0, 1.0))



# In[120]:


import matplotlib.pyplot as plt

plt.figure()
df.set_index("Mois",drop=True)
df[["Pigeon ramier","Mois"]].plot(color = 'r',linewidth=3,x="Mois")
df["Geai des chênes"].plot(color='g',linewidth=3.0,x="Mois")
plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.xticks(rotation=90)


# In[ ]:




