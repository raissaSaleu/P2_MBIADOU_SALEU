############### PROJET 2 #################

import pandas as pd
import numpy as np 
import random
import itertools
from collections import Counter 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

def poucentageValeursManquantes(data, column):
    ''' 
        Calcule le pourcentage moyen de valeurs manquantes
        dans une dataframe donnée par valeur unique d'une colonne donnée
        
        Parameters
        ----------------
        data   : dataframe contenant les données                 
        column : str
                 La colonne à analyser
        
        Returns
        ---------------
        Un dataframe contenant:
            - une colonne "column"
            - une colonne "Percent Missing" : contenant le pourcentage de valeur manquante pour chaque valeur de colonne    
    '''
    
    percent_missing = data.isnull().sum(axis=1) * 100 / len(data.columns)
    
    return pd.DataFrame({column: data[column], 'Percent Missing': percent_missing})\
                        .groupby([column])\
                        .agg('mean')
#------------------------------------------


def tauxRemplissageSelonCritere(data, criterion):
    ''' 
        Cette fonction prépare les données qui seront affichées pour montrer le taux de valeurs 
        renseignées/non renseignées liés à un critère "criterion" dans un dataframe "data" 
        
        Parameters
        ----------------
        - data : dataframe contenant les données
        - criterion: critère cible que l'on veut observer 
 
        Returns
        ---------------
        Un dataframe contenant:
            - une colonne "Percent Missing" : pourcentage de données non renseignées
            - une colonne "Percent Filled" : pourcentage de données renseignées
            - une colonne "Total": 100
            - une colonne qui prend la valeur de la variable critérion: la valeur du critère 
           
    '''    

    
    criterion_percent_missing_df = pd.DataFrame()

    for column, data_df in data.groupby(criterion):
        
        tmp_df = pd.DataFrame()

        tmp_df['Percent Missing'] = [poucentageValeursManquantes(data_df, 'Indicator Code')\
                                     ['Percent Missing'].mean()]
        
        tmp_df['Percent Filled'] = 100 - tmp_df['Percent Missing']
        tmp_df['Total'] = 100
        tmp_df[criterion] = column
        
        criterion_percent_missing_df = pd.concat([criterion_percent_missing_df, tmp_df])

    return criterion_percent_missing_df
#------------------------------------------


def plotTauxRemplissage(data, column, long, larg, part):
    ''' 
        Trace les proportions de valeurs remplies/manquantes pour chaque valeur unique
        dans la colonne sous forme de graphique à barres   horizontales empilées.
        
        Parameters
        ----------------
        data   : un dataframe avec : 
                   - une colonne column
                   - une colonne "Percent Missing" : pourcentage de données non renseignées
                   - une colonne "Percent Filled" : pourcentage de données renseignées
                   - une colonne "Total": 100
                                 
        column : str
                 La colonne devant servir d'axe des y pour le tracé
                                 
        long   : int 
                 La longueur de la figure
        
        larg   : int
                 La largeur de la figure
                                  
        
        Returns
        ---------------
        -
    '''
    
    data_to_plot = tauxRemplissageSelonCritere(data, column)\
                   .sort_values("Percent Missing", ascending=False).reset_index()

    TITLE_SIZE = 55
    TITLE_PAD = 100
    TICK_SIZE = 50
    TICK_PAD = 20
    LABEL_SIZE = 50
    LABEL_PAD = 50
    LEGEND_SIZE = 50
    Y_SIZE = 35

    sns.set(style="whitegrid")

    # Initialisation la figure matplotlib
    f, ax = plt.subplots(figsize=(long, larg))
    
    plt.title(part.upper()+": TAUX DE REMPLISSAGE DES DONNEES PAR " + column.upper(),
              fontweight="bold",
              fontsize=TITLE_SIZE, pad=TITLE_PAD)
    # Tracer les valeurs Total
    sns.set_color_codes("pastel")
    
    b = sns.barplot(x="Total", y=column, data=data_to_plot,label="non renseignées", color="b", alpha=0.3)
    b.set_xticklabels(b.get_xticks(), size = Y_SIZE)
    _, ylabels = plt.yticks()
    _, xlabels = plt.xticks()
    b.set_yticklabels(ylabels, size=Y_SIZE)


    # Tracer les valeurs Percent Filled
    sns.set_color_codes("muted")

    c = sns.barplot(x="Percent Filled", y=column, data=data_to_plot,label="renseignées", color="b")
    c.set_xticklabels(c.get_xticks(), size = Y_SIZE)
    c.set_yticklabels(ylabels, size=Y_SIZE)
    

    # Ajout d'une légende et une étiquette d'axe informative
    ax.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0, ncol=1, frameon=True,
             fontsize=LEGEND_SIZE)
    
    ax.set(ylabel=column,xlabel="Pourcentage de valeurs (%)")
    
    lx = ax.get_xlabel()
    ax.set_xlabel(lx, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")
    
    ly = ax.get_ylabel()
    ax.set_ylabel(ly, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")
    
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:2d}'.format(int(x)) + '%'))
    ax.tick_params(axis='both', which='major', pad=TICK_PAD)
    
    sns.despine(left=True, bottom=True)

    # Export de la figure
    plt.savefig('valeursRenseignees_'+column+'_'+part+'.png')
    
    # Affichage de la figure
    plt.show()

#------------------------------------------


def plotPourcentageValeursManquantes(filename, file_data_df):
    ''' 
        Cette fonction dessine un histogramme représentant le pourcentage de valeurs
        manquantes pour chaque colonne du fichier passé en paramètre
        
        Parameters
        ----------------
        - filename : le nom d'un fichier
        - file_data : le dataframe correspondant au fichier        
        
        Returns
        ---------------
        -    
    '''    
    
    sns.set(style="darkgrid")
    
    file_data_df.isnull().mean().plot.bar(figsize=(16,6))
    plt.ylabel('Valeurs manquantes (%)',fontsize=14)
    plt.xlabel('Colonnes', fontsize=14)
    plt.title('Pourcentage de valeurs manquantes du fichier '+filename, fontsize=20, fontweight='bold')
        
#------------------------------------------

def plotnombreDuplicationLignes(dataset):
    ''' 
        Cette fonction dessine une histogramme représentant le nombre de duplication de ligne trouvée dans
        chaque fichier du dataset
        
        Parameters
        ----------------
        Un dictionnaire avec:
        - clés : les noms des fichiers
        - valeurs : une liste contenant 2 valeurs: 
            - le dataframe correpondant à un fichier
            - une brève description du fichier        
        Returns
        ---------------
        Un dataframe contenant:
            - une colonne "Nom du fichier" : le nom du fichier
            - une colonne "Nb de duplications"   : le nombre de lignes dupliquées dans le fichier 
    '''    
    filenames = []
    nb_duplicates = []
    
    for filename, file_data in dataset.items():
        filenames.append(filename)
        nb_duplicates.append(file_data[0].duplicated().sum())    

    data_duplication_df = pd.DataFrame({'Nom du fichier':filenames, 
                                    'Nb de duplications':nb_duplicates})

    data_duplication_df.index += 1

    
    return data_duplication_df
#------------------------------------------

def tauxRemplissageIndicateurs (data, topic):
    ''' 
        Cette fonction calcule le taux de remplissage de données de tous les indicateurs associés à un Topic
        donné pour un jeu de données data
        
        Parameters
        ----------------
        - data : dataframe contenant les données
        - topic: le sujet concerné 
 
        Returns
        ---------------
        Un dataframe contenant:
            - une colonne "Code" : le code de l'indicateur 
            - une colonne "Decription" : description de l'indicateur
            - une colonne "Taux remplissage": le taux de remplissage de données pour cet indicateur dans data
           
    '''    
    
    data.query('Topic == @topic')
    
    indicators_code = data.query('Topic == @topic')['Indicator Code'].unique()
    descriptions = data.query('Topic == @topic')['Indicator Name'].unique()
    filling_rate = []
    
    for code in indicators_code:
        lines = data[data['Indicator Code'] == code]
        filling_rate.append((1-lines.isna().mean().mean())*100)

    filling_data_df = pd.DataFrame({'Code':indicators_code, 
                             'Description':descriptions,
                             'Taux remplissage': filling_rate
                            })
    filling_data_df = filling_data_df.sort_values(by=['Taux remplissage'],ascending=False)  
    
    filling_data_df.index += 1

    return filling_data_df

#------------------------------------------

def plotRepartitionTopics(data, long, larg):
    ''' 
        Plots a pie chart of the proportion of indicators per topic 
        in data
        
        Parameters
        ----------------
        data   : pandas dataframe with at least:
                     - a column "Topic"
                                  
        long   : int 
                 The length of the figure for the plot
        
        larg   : int
                 The width of the figure for the plot
                 
        Returns
        ---------------
        -
    '''
    
    TITLE_SIZE = 35
    TITLE_PAD = 60

    # Initialize the figure
    f, ax = plt.subplots(figsize=(long, larg))


    # Set figure title
    plt.title("RÉPARTITION DES INDICATEURS PAR TOPIC", fontweight="bold",
              fontsize=TITLE_SIZE, pad=TITLE_PAD)
       
    # Put everything in bold
    plt.rcParams["font.weight"] = "bold"


    # Create pie chart for topics
    a = data["Topic"].value_counts(normalize=True).plot(kind='pie', 
                                                        autopct=lambda x:'{:2d}'.format(int(x)) + '%', 
                                                        fontsize =30)
    # Remove y axis label
    ax.set_ylabel('')
    
    # Make pie chart round, not elliptic
    plt.axis('equal') 
    
    # Save the figure 
    plt.savefig('indicateursParTopic.png')
    
    # Display the figure
    plt.show()

#------------------------------------------


def dataframeChange (data_df, indicators):
    ''' 
        Cette fonction construit une nouvelle dataframe à partir de data_df et indicators
        dans laquelles les codes des indicateurs seront désormais des colonnes 
        
        Parameters
        ----------------
        data_df     : un data frame contenant les valeurs des indicateurs (dans "indicators") pour
                    tous les pays au fil des ans                                   
        
        indicators  : une liste d'indicateurs
        Returns
        ---------------
        Un dataframe contenant:
            - une colonne "Country Code" : code d'un pays 
            - une colonne "Country Name" : nom d'un pays
            - une colonne "Region" : la région d'un pays
            - une colonne "Year" : une année
            - une colonne pour chaque indicateur dans "indicators"
    '''
    
    data = data_df
    columns = ['Country Code', 'Country Name', 'Region', 'Year'] + indicators
    years = data.columns[4:-2]
    countries = data['Country Code'].unique()

    data_changed = []
    for year, country in itertools.product(years, countries):
        df_tmp = data[data["Country Code"]==country]
        out_tmp = [country, df_tmp['Country Name'].iloc[0], df_tmp['Region'].iloc[0], year]
        for ind in indicators:
            out_tmp.append(df_tmp[df_tmp['Indicator Code']==ind][year].values[0])
        data_changed.append(out_tmp)

    df_data_changed = pd.DataFrame(columns=columns, data=data_changed)
  
    return df_data_changed
#------------------------------------------



def normalisationMinMax(data_df, indicators):
    ''' 
        Cette fonction applique la normalisation min-max sur les valeurs
        de chaque indicateur par années
        
        Parameters
        ----------------
        data_df     : un data frame contenant les valeurs des indicateurs (dans "indicators") pour
                    tous les pays au fil des ans (final_historical_data_df ou final_projection_data_df)                                  
        
        indicators  : une liste d'indicateurs
        Returns
        ---------------
        Un dataframe contenant:
            - une colonne "Country Code" : code d'un pays 
            - une colonne "Country Name" : nom d'un pays
            - une colonne "Region" : la région d'un pays
            - une colonne "Year" : une année
            - une colonne pour chaque indicateur dans "indicators"
    '''
    
    df_data_changed = dataframeChange (data_df, indicators)
    data_norm_df = df_data_changed.copy()
    data_norm_df[indicators] = df_data_changed[indicators + ['Year']].groupby('Year').apply(lambda group:(group-group.min())/(group.max()-group.min()))

    
    return data_norm_df
#------------------------------------------

def plotIndicateursProjection(data_df, indicators, top_country):
    ''' 
        Cette fonction affiche l'évolution des 3 indicateurs de projection
        pour les 5 top pays
        
        Parameters
        ----------------
        data          : une dataframe contenant les données
                                 
        indicator     : un indicateur 

                                  
        
        Returns
        ---------------
        -
    '''
    df_data_changed = dataframeChange (data_df, indicators)
    top_country_year_df2 = df_data_changed[df_data_changed['Country Name'].isin(top_country)]
    
    for ind in indicators:
        fig,ax = plt.subplots(figsize=(8,6))
        fig.suptitle('PROJECTION: EVOLUTION DU NIVEAU D\'ETUDE ('+ind+')', fontsize=16, fontweight='bold')
        for label, df in top_country_year_df2.groupby('Country Name'):
            df.plot.line(x='Year', y=ind, ax=ax, label=label)
   
    #plt.show()

#------------------------------------------
    
    