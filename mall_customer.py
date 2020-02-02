from sklearn.preprocessing import StandardScaler
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from PIL import Image
import time


image = Image.open("walm1.jpg")
st.image(image ,width = 900, format="JPEG")#,caption="The GAME is ON!")
st.title("They Like, They Spend")

st.markdown("<h3></h3>",unsafe_allow_html=True)
st.markdown("<h3></h3>",unsafe_allow_html=True)
st.markdown("<h3></h3>",unsafe_allow_html=True)

@st.cache()
def read_data(filename):
    df1 = pd.read_csv(filename)
    return df1

df = pd.read_csv('Mall_Customers.csv')

st.header("Data~")
st.markdown("<h3></h3>",unsafe_allow_html=True)
st.write(df)

st.markdown("<h3></h3>",unsafe_allow_html=True)
st.markdown("<h3></h3>",unsafe_allow_html=True)
st.markdown("<h3></h3>",unsafe_allow_html=True)

st.header("Analysis~")
st.markdown("<h3></h3>",unsafe_allow_html=True)
st.write(df.describe())

st.markdown("<h3></h3>",unsafe_allow_html=True)
st.markdown("<h3></h3>",unsafe_allow_html=True)
st.markdown("<h3></h3>",unsafe_allow_html=True)

df.rename(index=str, columns={'Annual Income (k$)': 'Income',
                              'Spending Score (1-100)': 'Score'}, inplace=True)

# Let's see our data in a detailed way with pairplot
X = df.drop(['CustomerID', 'Gender'], axis=1)


#Comparative bivariate scatterplots
sns.pairplot(df.drop('CustomerID', axis=1), hue='Gender', aspect=1.5)
st.subheader('Comparative Analysis~')

st.markdown("<h4></h4>",unsafe_allow_html=True)

plot1 = st.pyplot()
fig = plt.figure(figsize=(8,5))

st.markdown("<h3></h3>",unsafe_allow_html=True)
st.markdown("<h3></h3>",unsafe_allow_html=True)
st.markdown("<h3></h3>",unsafe_allow_html=True)

#Choose algorithm
st.sidebar.header("Algorithms~")
grp = st.sidebar.multiselect("Choose algorithm(s) to apply:",['K-Means','Agglomerative Hierarchical Clustering'])\
    #,'Density Based Clustering','Mean Shift Algorithm',])


# K-Means Clustering
if("K-Means" in grp):
    clusters=[]
    st.header("Elbow Point~")
    #st.markdown("<h3></h3>", unsafe_allow_html=True)

    for i in range(1, 11):
        km = KMeans(n_clusters=i).fit(X)
        clusters.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.lineplot(x=list(range(1, 11)), y=clusters, ax=ax)
    ax.set_title('Searching for Elbow')
    ax.set_xlabel('Clusters')
    ax.set_ylabel('Inertia')


    # Annotate arrow
    ax.annotate('Possible Elbow Point', xy=(3, 140000), xytext=(3, 50000), xycoords='data',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2))

    ax.annotate('Possible Elbow Point', xy=(5, 80000), xytext=(5, 150000), xycoords='data',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2))

    elbo_plot = st.pyplot()

    st.markdown("<h3></h3>", unsafe_allow_html=True)
    st.markdown("<h3></h3>", unsafe_allow_html=True)
    st.markdown("<h3></h3>", unsafe_allow_html=True)

    #fitting k-means in different values of K

    st.sidebar.subheader("K-Value")
    clust = st.sidebar.slider("Choose number of Clusters:",2,10,5,1,key='K-Meanslider')

    km_header = st.empty()
    km_plot = st.empty()
    progress1 = st.empty()
    sw_header = st.empty()
    sw_plot = st.empty()


    def k_means(n_clust):
        kmean = KMeans(n_clusters=n_clust).fit(X)
        X['Labels'] = kmean.labels_
        st.write(X)
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.markdown("<h3></h3>", unsafe_allow_html=True)

        plt.figure(figsize=(10,8))
        sns.scatterplot(X['Income'], X['Score'], hue=X['Labels'], markers=True, size=X['Labels'],
                        palette=sns.color_palette('hls', n_clust))


        #to give numbering to each cluster


        for label in X['Labels']:
            plt.annotate(label,
                         (X[X['Labels'] == label]['Income'].mean(),
                         X[X['Labels'] == label]['Score'].mean()),
                         horizontalalignment='center',
                         verticalalignment='center',
                         size=20, weight='bold',
                         color= 'black')
        # #                  #backgroundcolor=customPalette[i])



        km_header.header('Cluster Plot~')
        km_plot.pyplot()
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.markdown("<h3></h3>", unsafe_allow_html=True)

        #plt.show()

        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(121)
        sns.swarmplot(x='Labels', y='Income', data=X, ax=ax)
        ax.set_title('Labels According to Annual Income')

        ax = fig.add_subplot(122)
        sns.swarmplot(x='Labels', y='Score', data=X, ax=ax)
        ax.set_title('Labels According to Scoring History')

        #plt.show()

        sw_header.header('Swarm Plot~')
        #st.markdown("<h3></h3>", unsafe_allow_html=True)
        sw_plot.pyplot()

        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.markdown("<h3></h3>", unsafe_allow_html=True)

    k_means(clust)

    #the slider cannot be externally controlled when it is in sidebar
    #AttributeError: 'function' object has no attribute 'slider' (since sidebar() is a function


    def move_slider(position):
        progress1.slider('Number of Clusters', 2, 10, position, 1)

    if(st.sidebar.button("Simulate",key='K-Means')):
        #slider1 = st.sidebar.empty()
        for i in range(2,11):
            #slider1.sidebar.slider('Choose number of Clusters:',2,10,i,1)
            move_slider(i)
            k_means(i)





#Agglomerative Clustering
if('Agglomerative Hierarchical Clustering' in grp):
    from sklearn.cluster import AgglomerativeClustering

    #agglom =
    st.sidebar.header("Number of Clusters~")
    clust2 = st.sidebar.slider("Choose number of Clusters:", 2, 10, 5, 1,key='Agglom cluster sliderstreamlit run ')

    ag_header = st.empty()
    ag_plot = st.empty()
    progress2 = st.empty()
    dendro_header1 = st.empty()
    dendro_plot1 = st.empty()
    dendro_header2 = st.empty()
    dendro_plot2 = st.empty()

    def agglom_clust(n_clust):
        agglom = AgglomerativeClustering(n_clusters=n_clust, linkage='average').fit(X)
        X['Labels'] = agglom.labels_
        # st.write(X)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(X['Income'], X['Score'], hue=X['Labels'], markers=True, size=X['Labels'],
                        palette=sns.color_palette('hls', n_clust))
        plt.title('Agglomerative Clustering')

        # to give numbering to each cluster
        # for label in X['Labels']:
        #     plt.annotate(label,
        #                  (X[X['Labels'] == label]['Income'].mean(),
        #                   X[X['Labels'] == label]['Score'].mean()),
        #                  horizontalalignment='center',
        #                  verticalalignment='center',
        #                  size=20, weight='bold',
        #                  color='black')
                #backgroundcolor=customPalette[i])

        ag_header.header('Cluster Plot~')
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        ag_plot.pyplot()
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.markdown("<h3></h3>", unsafe_allow_html=True)


        #Dendrograph

    from scipy.cluster import hierarchy
    from scipy.spatial import distance_matrix

    dist = distance_matrix(X, X)
    Z = hierarchy.linkage(dist, 'complete')
    plt.figure(figsize=(8, 12))
    dendro = hierarchy.dendrogram(Z, leaf_rotation=0, leaf_font_size=12, orientation='right')
    dendro_header1.header("Complete Linkage Dendrogram~")
    dendro_plot1.pyplot()
    Z = hierarchy.linkage(dist, 'average')
    plt.figure(figsize=(8, 12))
    dendro1 = hierarchy.dendrogram(Z, leaf_rotation=0, leaf_font_size=12, orientation='right')
    dendro_header2.header("Average Linkage Dendrogram~")
    st.markdown("<h3></h3>", unsafe_allow_html=True)
    dendro_plot2.pyplot()
    st.markdown("<h3></h3>", unsafe_allow_html=True)
    st.markdown("<h3></h3>", unsafe_allow_html=True)
    st.markdown("<h3></h3>", unsafe_allow_html=True)
    # plt.show()



    def move_slider2(position):
        progress2.slider('Number of Clusters', 2, 10, position, 1, key='Agglomprogressslider')

    if(st.sidebar.button("Simulate",key='Agglom')):
        #slider1 = st.sidebar.empty()
        for i in range(2,11):
            #slider1.sidebar.slider('Choose number of Clusters:',2,10,i,1)
            move_slider2(i)
            agglom_clust(i)


    agglom_clust(clust2)


