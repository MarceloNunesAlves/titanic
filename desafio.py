import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import xgboost as xgb


def preparar_dataframe(df):
    # [i for i, v in enumerate(df) if 'male' in v]
    # Converter sexo
    df['Sexo'] = df.Sex.map({'female': 0, 'male': 1})

    # Converter se tem irmão ou conjugê
    df['Tem_irmao_conjuge'] = df.SibSp.map(lambda x: 1 if x > 0 else 0)

    # Converter se tem filhos
    df['Tem_filhos'] = df.Parch.map(lambda x: 1 if x > 0 else 0)

    # Separa a coluna de portão de entrada
    #df['entrada_Cherbourg'] = df.Embarked.map(lambda x: 1 if x == 'C' else 0)
    #df['entrada_Southampton'] = df.Embarked.map(lambda x: 1 if x == 'S' else 0)
    #df['entrada_Queenstown'] = df.Embarked.map(lambda x: 1 if x == 'Q' else 0)

    # Separa as Classes sociais
    df['primeira'] = df.Pclass.map(lambda x: 1 if x == 1 else 0)
    df['segunda'] = df.Pclass.map(lambda x: 1 if x == 2 else 0)
    df['terceira'] = df.Pclass.map(lambda x: 1 if x == 3 else 0)

    #Idade
    df['idade_0_5'] = df.Age.map(lambda x: 1 if x >= 0 or x < 5 else 0)
    df['idade_5_10'] = df.Age.map(lambda x: 1 if x >= 5 or x < 10 else 0)
    df['idade_10_15'] = df.Age.map(lambda x: 1 if x >= 10 or x < 15 else 0)
    df['idade_15_20'] = df.Age.map(lambda x: 1 if x >= 15 or x < 20 else 0)
    df['idade_20_25'] = df.Age.map(lambda x: 1 if x >= 20 or x < 25 else 0)
    df['idade_25_30'] = df.Age.map(lambda x: 1 if x >= 25 or x < 30 else 0)
    df['idade_30_35'] = df.Age.map(lambda x: 1 if x >= 30 or x < 35 else 0)
    df['idade_35_40'] = df.Age.map(lambda x: 1 if x >= 35 or x < 40 else 0)
    df['idade_40_45'] = df.Age.map(lambda x: 1 if x >= 40 or x < 45 else 0)
    df['idade_45_50'] = df.Age.map(lambda x: 1 if x >= 45 or x < 50 else 0)
    df['idade_50_55'] = df.Age.map(lambda x: 1 if x >= 50 or x < 55 else 0)
    df['idade_55_60'] = df.Age.map(lambda x: 1 if x >= 55 or x < 60 else 0)
    df['idade_60_70'] = df.Age.map(lambda x: 1 if x >= 60 or x < 70 else 0)
    df['idade_70'] = df.Age.map(lambda x: 1 if x >= 70 else 0)
    df['idade_n_definida'] = df.Age.map(lambda x: 1 if pd.isnull(x) else 0)

    # Remove colunas desnecessarias
    df.drop(columns=['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
            inplace=True)

    if 'Survived' in df.columns:
        df.drop(columns=['Survived'], inplace=True)

    # Substitui o NaN por 0
    # df = df.fillna(0)

    return df

def teste_real(resultado, validacao_marcacoes):
    resultado = np.array([resultado])
    validacao_marcacoes = np.array(validacao_marcacoes)

    acertos = np.equal(resultado, validacao_marcacoes)

    total_de_acertos = np.sum(acertos)
    total_de_elementos = len(validacao_marcacoes)

    taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

    return taxa_de_acerto

def data_predict(modelo, validacao_dados):
    resultado = modelo.predict(validacao_dados)
    return resultado

#file = np.genfromtxt('train.csv', delimiter=',', dtype=str, skip_header=True, usecols=[1,2,5,6,7,8])
file = pd.read_csv('train.csv')

dados = file
label = dados['Survived']

# Prepara treino/teste
dados = preparar_dataframe(dados)
print(dados)

train, teste, label, label_teste = train_test_split(dados, label, test_size=0.20, random_state=0)

#modelo = AdaBoostClassifier(random_state=0)
#modelo = GaussianNB()
modelo = xgb.XGBClassifier()


xgb_preds = []
K=10
kf = KFold(n_splits=K, shuffle=False, random_state=None)
for train_index, test_index in kf.split(train):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = train.iloc[train_index], train.iloc[test_index]
    y_train, y_test = label.iloc[train_index], label.iloc[test_index]

    modelo.fit(X_train, y_train)

    # Prepara treino
    resultado = data_predict(modelo, X_test)
    print("Taxa de acerto na validação cruzada: {0}".format(teste_real(resultado, y_test)))


# Prepara treino
modelo.fit(train, label)
resultado = data_predict(modelo, teste)
print("Taxa de acerto no mundo real: {0}".format(teste_real(resultado, label_teste)))

# Gerar resultado
teste = pd.read_csv('test.csv')
# Separar os ID de passageiros
ids = teste['PassengerId']

teste = preparar_dataframe(teste)
resultado = data_predict(modelo, teste)
#Survived
output_file = pd.concat([ids, pd.Series(data=resultado,name='Survived')], axis=1)
output_file.to_csv('result_file.csv', index=False)