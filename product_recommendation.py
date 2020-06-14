import heapq
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from surprise.model_selection import train_test_split, cross_validate
from surprise import SVD, Dataset, Reader, SVDpp, SlopeOne, NMF, KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore, \
    BaselineOnly, CoClustering, NormalPredictor, accuracy
import pprint
pp = pprint.PrettyPrinter()

class ProductRecommendation():
    def __init__(self):
        self.sales_list_df = self.getSalesList()
        self.product_df = self.ProductList()
        self.lower_rating = self.sales_list_df['sum_quantity'].min()
        self.upper_rating = self.sales_list_df['sum_quantity'].max()
        self.data = self.LoadDataset()
        self.train_set, self.test_set = train_test_split(self.data, test_size=0.20)
        self.algo = SVDpp()
        self.algo.fit(self.train_set)
        pred = self.algo.test(self.test_set)
        # Test score
        score = accuracy.rmse(pred)

    def EvaluateAllModels(self):
        """
                         test_rmse   fit_time  test_time
        Algorithm
        SVDpp             0.965824   9.401286   0.151476
        SVD               0.967286   1.474139   0.062471
        BaselineOnly      0.972408   0.108964   0.057277
        NMF               0.992677   4.073005   0.171846
        KNNWithZScore     1.001898   0.620192   0.083341
        KNNWithMeans      1.002924   0.489803   0.078121
        SlopeOne          1.006664  19.091191   1.275676
        KNNBaseline       1.007437   0.890452   0.088495
        KNNBasic          1.016717   0.432159   0.072929
        NormalPredictor   1.253265   0.041646   0.078105
        CoClustering      1.828291   3.020921   0.052071
        :return: test_rmse sonucu en düşük olan alınır.
        """
        benchmark = []
        # Iterate over all algorithms
        for algorithm in [SVD(), SVDpp(), SlopeOne(), NMF(), NormalPredictor(), KNNBaseline(), KNNBasic(),
                          KNNWithMeans(), KNNWithZScore(), BaselineOnly(), CoClustering()]:
            # Perform cross validation
            results = cross_validate(algorithm, self.data, measures=['RMSE'], cv=3, verbose=False)

            # Get results & append algorithm name
            tmp = pd.DataFrame.from_dict(results).mean(axis=0)
            tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
            benchmark.append(tmp)

        result = pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse')
        print(result)

        return result

    def Visualize(self):
        # create a figure and axis
        fig, ax = plt.subplots()
        data = self.sales_list_df.groupby('ProductTypeID')['sum_quantity'].count().clip(upper=50)
        # scatter the sepal_length against the sepal_width
        ax.scatter(data.index, data.values, color='teal')
        # set a title and labels
        ax.set_title('Sales Quantity Per Product')
        ax.set_xlabel('ProductTypeID')
        ax.set_ylabel('sum_quantity')

        plt.show()

    def getSalesList(self):
        sales_list = pd.read_csv("sales_list.csv", sep=',',  encoding='iso8859_9')
        sales_list_df = pd.DataFrame(sales_list, columns=['user_id', 'ProductTypeID', 'sum_quantity'])
        return sales_list_df

    def ProductList(self):
        product = pd.read_csv("product_list.csv", sep=',')
        product_df = pd.DataFrame(product, columns = ['ProductTypeID', 'product_name_options'])
        return product_df

    def LoadDataset(self):
        print("Review range {0} to {1}".format(self.lower_rating, self.upper_rating))
        reader = Reader(rating_scale=(self.lower_rating, self.upper_rating))
        data = Dataset.load_from_df(self.sales_list_df, reader)
        return data

    def recommend_product(self,user_id,num_recommendations):
        # Get a list of all product_ids
        iids = self.sales_list_df['ProductTypeID'].unique()
        # Get list of product_ids that user_id has bought.
        iid_user = self.sales_list_df[self.sales_list_df['user_id'] == user_id]
        print('User {0} has already bought {1} the product.'.format(user_id, iid_user.shape[0]))
        # user and bought product information
        user_full = (iid_user.astype(int).merge(self.product_df, how='left', left_on='ProductTypeID',
                                                 right_on='ProductTypeID').sort_values(['sum_quantity'], ascending=False))

        print(user_full)
        # Remove the iids that user has bought from the list of all product ids
        iids_to_pred = np.setdiff1d(iids,user_full['ProductTypeID'])
        print('Recommend the highest rated {0} rating products not yet rated.'.format(num_recommendations))
        test_rating = [[user_id, iid, self.upper_rating] for iid in iids_to_pred]
        predictions = self.algo.test(test_rating)
        pred_ratings = np.array([pred.est for pred in predictions])
        i_max = heapq.nlargest(num_recommendations, range(len(pred_ratings)), pred_ratings.take)
        iid = iids_to_pred[i_max]
        iid_df = pd.DataFrame({'ProductTypeID':iid})
        recommended_full = (iid_df.astype(int).merge(self.product_df, how='left', left_on='ProductTypeID', right_on='ProductTypeID').iloc[:num_recommendations, :])

        # Get and sort the user's predictions
        print("Recommend top products for user {0} ".format(user_id))
        user_id_np = np.repeat(user_id, num_recommendations)
        recommended_full['user_id'] = user_id_np
        print(recommended_full)
        recommended_full.to_csv('recommended_result.csv', header=True, index=False, sep=',', encoding='iso8859_9')
        print("done")


    def run(self):
        print("start")
        self.recommend_product(201,5)
        self.EvaluateAllModels()
        self.Visualize()

