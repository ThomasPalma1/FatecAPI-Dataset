import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from flask import Blueprint
import os

recommendationRoute = Blueprint("recommendationRoute", __name__)


def making_it_happen():
    #Importação dos serviços de bancos categorizados
    path_dir = os.path.dirname(os.path.abspath(__file__))

    # Importação dos serviços de bancos categorizados
    services = pd.read_csv(os.path.join(path_dir, "consolidado_Servicos_categoria_id.csv"), encoding="utf-8", sep=";",
                           on_bad_lines='skip')
    # Importação dos dados referentes às notas de serviços de bancos
    ratings = pd.read_csv(os.path.join(path_dir, "service_ratings_category.csv"), encoding="latin-1", sep=";",
                          on_bad_lines='skip')
    services = services[['Cnpj','BancoID','RazaoSocial','ValorMaximo','ListaServiço','Categoria']]
    services.rename(columns = {'Cnpj':'cnpj','BancoID':'service_id','RazaoSocial':'razaosocial','ValorMaximo':'valormaximo','ListaServico':'listaservico','Categoria':'categoria'}, inplace=True)
    #Quantidade de ratings por usuários
    ratings['user_id'].value_counts()
    #serviços com mais de 50 avaliações
    qtdavaliacao = ratings['user_id'].value_counts()>50
    #Quantidade de avaliadores com de 50 serviços avaliados
    qtdusers = qtdavaliacao[qtdavaliacao].index
    print(qtdusers.shape)
    qtdusers
    #Trazendo notas de usuários que avaliaram mais de 50 serviços
    ratings = ratings[ratings['user_id'].isin(qtdusers)]
    #Juntando tabelas
    rating_with_services= ratings.merge(services, on='service_id')
    rating_with_services.head()
    #Quantidade de rating de serviços
    number_rating = rating_with_services.groupby('ListaServiço')['service_rating'].count().reset_index()
    number_rating
    #Renomeando coluna
    number_rating.rename(columns={'service_rating':'number_of_ratings'}, inplace=True)
    number_rating
    # Junbtando tabela de serviços com os ratings e tabela de quintadad de ratings por serviço
    final_rating = rating_with_services.merge(number_rating, on='ListaServiço')
    final_rating
    #Filtrar serviços que tenham pelo menos 50 avaliações
    final_rating = final_rating[final_rating['number_of_ratings'] >=50]
    final_rating.shape
    #Eliminar duplicatas de usuarios que avaliaram mesmo serviço varias vezes
    final_rating.drop_duplicates(['user_id','ListaServiço'],inplace=True)
    final_rating.shape
    final_rating
    service_pivot = final_rating.pivot_table(columns='user_id',index='ListaServiço',values='service_rating')
    service_pivot
    service_pivot.shape
    from sqlalchemy.sql.base import InPlaceGenerative
    service_pivot.fillna(0,inplace=True)
    service_pivot
    from scipy.sparse import csr_matrix
    service_sparce = csr_matrix(service_pivot)
    from sklearn.neighbors import NearestNeighbors
    model = NearestNeighbors(algorithm='brute')
    model.fit(service_sparce)
    service_pivot.head()
    return model, service_pivot


@recommendationRoute.route('/recommend/<servico>')
def recommend(servico):
    model, service_pivot = making_it_happen()
    distances, suggestions = model.kneighbors(service_pivot.filter(items=[servico], axis=0).values.reshape(1, -1))
    for i in range(len(suggestions)):
        suggestion = service_pivot.index[suggestions[i]]
        suggestion = " ".join(suggestion)
        print(suggestion)
        return  suggestion
