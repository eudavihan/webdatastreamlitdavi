# Importação das bibliotecas utilizadas no código
import streamlit as st
import pandas as pd
import sqlalchemy as db
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import numpy as np

#############################################################################################################
# Conexão ao banco de dados e carregamento dos dados ao código

# Criar uma engine para o banco de dados SQLite
conn = st.connection("mydb", type="sql")

# Definir uma função para carregar dados do banco de dados
@st.cache_data
def load_data():
    query = "SELECT * FROM 'HOF Players'"
    df = conn.query(query)
    return df

# Carregar os dados
df = load_data()


#############################################################################################################
# Definir grupos de posições e cores
posicoes_ofensivas = ['QB', 'WR', 'G', 'C', 'RB', 'T', 'TE']  
posicoes_defensivas = ['DB', 'DE', 'DT', 'LB']  
posicoes_times_especiais = ['K', 'PR']  

@st.cache_data
def categorizar_e_atribuir_cor(posicao):
    if posicao in posicoes_ofensivas:
        return 'Ataque', 'red'
    elif posicao in posicoes_defensivas:
        return 'Defesa', 'blue'
    elif posicao in posicoes_times_especiais:
        return 'Times Especiais', 'green'

# Aplicar categorização e cores ao DataFrame
df['Categoria'], df['Color'] = zip(*df['Position'].apply(categorizar_e_atribuir_cor))

cores_categorias = {
    'Ataque': 'red',
    'Defesa': 'blue',
    'Times Especiais': 'green'
}

# Contagem de posições
contagem_posicoes = df['Position'].value_counts().reset_index()
contagem_posicoes.columns = ['Position', 'Player Count']
contagem_posicoes['Categoria'] = contagem_posicoes['Position'].apply(lambda pos: categorizar_e_atribuir_cor(pos)[0])

#############################################################################################################
# Título, logo e descrição no Streamlit

# Função para exibir o logo no Streamlit
st.image('https://upload.wikimedia.org/wikipedia/en/thumb/7/71/Pro_Football_Hall_of_Fame_logo.svg/1024px-Pro_Football_Hall_of_Fame_logo.svg.png')  # Ajustar o caminho ou usar URL online

# Função para mostrar um título e descrição do aplicativo no Streamlit
st.title("Hall da Fama da NFL")
st.write("Essa aplicação demonstra dados dos últimos 100 jogadores homenageados ao Hall da Fama da NFL.")

#############################################################################################################
# Criação de visualização de dados no Streamlit

# Seleção de posições para o gráfico de barras
posicoes_selecionadas = st.multiselect(
    "Selecione posições para incluir no gráfico de barras:", 
    df['Position'].unique(), 
    df['Position'].unique()
)

# Filtragem de dados para o gráfico de barras
contagem_posicoes_filtradas = contagem_posicoes[contagem_posicoes['Position'].isin(posicoes_selecionadas)]
grafico_barras_filtrado = px.bar(
    contagem_posicoes_filtradas, 
    x='Position', y='Player Count', color='Categoria',
    color_discrete_map=cores_categorias,
    title="Jogadores do Hall da Fama por Posição",
    labels={'Player Count': 'Nº de jogadores', 'Position': 'Posições'}
)
st.plotly_chart(grafico_barras_filtrado)

st.markdown("---")

# Filtro de faixa de Temporadas All-Pro
min_todas_pro = int(df['All Pro Seasons'].min())
max_todas_pro = int(df['All Pro Seasons'].max())
faixa_selecionada = st.slider(
    "Selecione a faixa de Nº de Temporadas All-Pro para incluir:", 
    min_todas_pro, max_todas_pro, (min_todas_pro, max_todas_pro)
)

# Filtragem de dados para o gráfico de dispersão
contagem_todas_pro_filtradas = df[(df['All Pro Seasons'] >= faixa_selecionada[0]) & (df['All Pro Seasons'] <= faixa_selecionada[1])]
grafico_dispersao_filtrado = px.scatter(
    contagem_todas_pro_filtradas.groupby('All Pro Seasons').size().reset_index(name='Player Count'), 
    x='All Pro Seasons', y='Player Count', color='All Pro Seasons',
    color_continuous_scale='Viridis',
    title="Relação entre Nº de Temporadas All-Pro e Nº de Homenageados ao Hall da Fama",
    labels={'All Pro Seasons': 'Nº de temporadas All-Pro', 'Player Count': 'Nº de Jogadores'},
    size='Player Count', size_max=40
)
st.plotly_chart(grafico_dispersao_filtrado)

st.markdown("---")

# Filtro de categoria
categorias_selecionadas = st.multiselect(
    "Selecione as categorias para incluir:", 
    df['Categoria'].unique(), 
    ['Ataque', 'Defesa', 'Times Especiais']
)

# Filtragem de dados para o gráfico de dispersão 3D
df_filtrado = df[df['Categoria'].isin(categorias_selecionadas)]
grafico_dispersao_3d_filtrado = px.scatter_3d(
    df_filtrado, x='All Pro Seasons', y='Pro Bowl Seasons', z='Games Played',
    color='Categoria',
    color_discrete_map=cores_categorias,
    title="Relação entre Temporadas All-Pro, Temporadas Pro Bowl e Total de jogos",
    labels={'All Pro Seasons': 'Temporadas All-Pro', 'Pro Bowl Seasons': 'Temporadas Pro Bowl', 'Games Played': 'Total de Jogos'},
    size_max=10
)

X = df_filtrado[['All Pro Seasons', 'Pro Bowl Seasons']]
X = sm.add_constant(X)  # Adicionar constante para o intercepto
y = df_filtrado['Games Played']
modelo = sm.OLS(y, X).fit()

x_range = np.linspace(X['All Pro Seasons'].min(), X['All Pro Seasons'].max(), 100)
y_range = np.linspace(X['Pro Bowl Seasons'].min(), X['Pro Bowl Seasons'].max(), 100)
x_grid, y_grid = np.meshgrid(x_range, y_range)
z_pred = modelo.params[0] + modelo.params[1] * x_grid + modelo.params[2] * y_grid

grafico_dispersao_3d_filtrado.add_trace(
    go.Surface(x=x_grid, y=y_grid, z=z_pred, colorscale='Viridis', opacity=0.5, name='Superfície de Regressão')
)

grafico_dispersao_3d_filtrado.update_layout(
    legend=dict(
        x=0,
        y=1,
        traceorder='normal',
        font=dict(
            size=10,
        ),
    ),
    scene=dict(
        xaxis=dict(title='Temporadas All-Pro'),
        yaxis=dict(title='Temporadas Pro Bowl'),
        zaxis=dict(title='Total de Jogos')
    )
)

st.plotly_chart(grafico_dispersao_3d_filtrado)

st.markdown("---")

# Exibir o DataFrame
st.write(df)

# Adicionar uma checkbox para mostrar os dados brutos
if st.checkbox("Mostrar dados brutos"):
    st.subheader("Dados brutos")
    st.write(df)
