'''
Who Buys Tickets at Enigma Stadium?
Enigma Stadium has some data about their customers. Each row represents the information about an individual who either did or didn't buy tickets, as indicated by 1 or 0 in the will_buy_ticket column. The other columns are demographic data about who the individuals are, where they live, and which sports they enjoy.

Enigma Stadium eventually wants to use this to build a model for targeting their marketing campaigns, but first they want you to look through the data and make sure that:
    - The data is representative of the real world (or highlight where it's not).
    - A set of features is generated for the data that follows ML and data cleaning best practices.

For this interview, please consider all online resources and any kind of external knowledge fair game.

''' 
import pdb
import pandas as pd
pd.set_option("display.max_columns", None)

import requests
url = 'https://gist.githubusercontent.com/adkatrit/99ae692b007a27d686325a92af533484/raw/b720c68af26d2d71d4fbb44e13124950a8df6b5a/enigma_stadium.csv'
r = requests.get(url, allow_redirects=True)
open('enigma_stadium.csv', 'wb').write(r.content)

data = pd.read_csv("enigma_stadium.csv")

pdb.set_trace()
data.head()
