
'''
Modeling the behavior of a perishable product
You are trying to model the decay of a perishable produce item from data. You are given the item's recorded sales, shipments, and a noisy signal of its beginning-of-day inventory over a consecutive N day period.

== Instructions ==
Please fill in the class PerishableProductModel (or write your own class). This class has two external methods, fit and predict.
    - fit fits a decay model to historical inventory data
    - predict predicts future inventories using the decay model on a set of sales and shipment data

== Details ==
The data follows the following physical process. Each day:

    - Inventory is recorded.
    - A shipment arrives in the store.
    - Items are sold.
    - Some items that are left in stock perish according to your model and are no longer sellable.
    -Any items that are left are available to sell the following day.

== More Details ==
    - Items are sold FIFO (first-in-first-out)
    - The inventory observations have been corrupted with zero mean Gaussian random noise and then rounded to the nearest positive integer.
    - The first inventory observation is a true value, and all of the items are 1 day old.

== Some Modeling Options:==
We can simplify our problem into two possible ways that items decay. You will have to decide which of these models of perishability best fits our data, and fit its parameters to the data we observe.

Model 1: Percentage Based

In a percentage based model, we are trying to estimate a single float parameter p that tells us what fraction of items go bad in a day. This might be a good model for something like potatoes, which last a long time but sometimes get dropped on the floor.

Model 2: Age Based

In an age-based model, we are trying to estimate a single integer parameter N. This parameter tells us how long an item lasts after being shipped into a store. We can assume in this case that all items come in with the same remaining shelf life.


== Explanation of Data: == 
We have two sets of 20 days of consecutive observational data for the same item.
     - inv_observation_{1..2}: Noisy observations of inventory. inv_observation[i] is the number of items seen on the morning of day i.
     - shipments_{1..2}: Recorded shipments into the store. shipments[i] is the number of new items that arrive in the store on day i.
     - sales_{1..2}: Recorded sales out of the store. sales[i] is the number of sales out of the store on day i.

'''

import sys
import pdb

import matplotlib.pyplot as plt
import numpy as np

inv_observation_1 = np.array([10, 26, 5, 5, 15, 18, 27, 15, 10, 18, 33, 17, 20, 23, 25, 0, 5, 8, 10, 11])
shipments_1 = np.array([16, 0, 6, 14, 18, 19, 8, 8, 15, 16, 10, 9, 19, 1, 0, 5, 10, 2, 3, 9])
sales_1 = np.array([5, 20, 0, 4, 13, 6, 29, 10, 5, 4, 1, 7, 13, 6, 15, 0, 1, 4, 10, 3])

inv_observation_2 =  np.array([10, 14, 11, 0, 17, 28, 18, 6, 25, 5, 19, 19, 26, 11, 24, 13, 11, 18, 16, 12])
shipments_2 = np.array([8, 12, 9, 16, 15, 6, 6, 14, 9, 7, 13, 12, 1, 13, 6, 4, 8, 14, 3, 16])
sales_2 = np.array([6, 4, 11, 11, 14, 4, 7, 7, 2, 14, 7, 0, 0, 7, 4, 0, 2, 11, 14, 0])



class PerishableProductModel:
  def __init__(self) -> None:
    pass

  def fit(self,
          true_sales: np.array,
          true_shipments: np.array,
          noisy_inventories: np.array) -> None:
    """Fit your model to the data"""
    sarima(self.decay, self.decay[-N])  
      
    pass
  
  def predict(self,
              sales: np.array,
              shipments: np.array,
              starting_inventory: float) -> np.array:
    """Predict inventories using your model"""
    pass

  def pre_proc(self,
          true_sales: np.array,
          true_shipments: np.array,
          noisy_inventories: np.array) -> None:

          self.decay = ((self.noisy_inventories - self.sales + self.shipments_1) - self.noisy_inventories.shift(-1))/slef.noisy_inventories
          return 
