!pip install --upgrade ibmos2spark
!pip install --upgrade pixiedust
!pip install --upgrade watson-machine-learning-client
import pixiedust
df = pixiedust.sampleData('https://3999016.app.netsuite.com/core/media/media.nl?id=732509&c=3999016&h=82edd9aa34896e0e2b2b&_xt=.csv')
display(df)
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.clustering import KMeansModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors
product_cols = ['Baby Food', 'Diapers', 'Formula', 'Lotion', 'Baby wash', 'Wipes', 'Fresh Fruits', 'Fresh Vegetables', 'Beer', 'Wine', 'Club Soda', 'Sports Drink', 'Chips', 'Popcorn', 'Oatmeal', 'Medicines', 'Canned Foods', 'Cigarettes', 'Cheese', 'Cleaning Products', 'Condiments', 'Frozen Foods', 'Kitchen Items', 'Meat', 'Office Supplies', 'Personal Care', 'Pet Supplies', 'Sea Food', 'Spices']
# Here we get the customer ID and the products they purchased
df_filtered = df.select(['CUST_ID'] + product_cols)
from six import iteritems
def get_product_counts_for_customer(cust_id):
    cust = df_customer_products.filter('CUST_ID = {}'.format(cust_id)).take(1)
    fields = []
    values = []
    for row in customer:
        for product_col in product_cols:
            field = 'sum({})'.format(product_col)
            value = row[field]
            fields.append(field)
            values.append(value)
    return (fields, values)
		def get_cluster_from_watson_ml(fields, values):
    scoring_payload = {'fields': fields, 'values': [values]}
    predictions = client.deployments.score(scoring_endpoint, scoring_payload)   
    return predictions['values'][0][len(product_cols)+1]
		# This function gets the most popular clusters in the cell by grouping by the cluster column

def get_popular_products_in_cluster(cluster):
    
    df_cluster_products = df_customer_products_cluster.filter('cluster = {}'.format(cluster))
    
    df_cluster_products_agg = df_cluster_products.groupby('cluster').sum()
    
    row = df_cluster_products_agg.rdd.collect()[0]
    

    
    items = []
    
    for product_col in product_cols:
        
        field = 'sum(sum({}))'.format(product_col)
        
        items.append((product_col, row[field]))
        
    sortedItems = sorted(items, key=lambda x: x[1], reverse=True) # Sort by score
    
    popular = [x for x in sortedItems if x[1] > 0]
    
    return popular
		from pyspark.sql.functions import desc

def get_recommendations_by_cluster(cluster, purchased_quantities):
    
    # Existing customer products
    
    print('PRODUCTS ALREADY PURCHASED/IN CART:')
    
    customer_products = []
    
    for i in range(0, len(product_cols)):
        
        
        
        if purchased_quantities[i] > 0:
                      
            
    df_customer_products = sc.parallelize(customer_products).toDF(["PRODUCT","COUNT"])
    
    df_customer_products.show()
    
    # Get popular products in the cluster
    
    print('POPULAR PRODUCTS IN CLUSTER:')
    
    cluster_products = get_popular_products_in_cluster(cluster)
    
    df_cluster_products = sc.parallelize(cluster_products).toDF(["PRODUCT","COUNT"])
    
    df_cluster_products.show()
    
    # Filter out products the user has already purchased
    
    print('RECOMMENDED PRODUCTS:')
    
    df_recommended_products = df_cluster_products.alias('cl').join(df_customer_products.alias('cu'), df_cluster_products['PRODUCT'] == df_customer_products['PRODUCT'], 'leftouter')
    
    df_recommended_products = df_recommended_products.filter('cu.PRODUCT IS NULL').select('cl.PRODUCT','cl.COUNT').sort(desc('cl.COUNT'))
    
    df_recommended_products.show(10)
		def get_recommendations_for_shopping_cart(products, quantities):
    
    fields = []
    
    values = []
    
    for product_col in product_cols:
        
        field = 'sum({})'.format(product_col)
        
        if product_col in products:
            
            value = quantities[products.index(product_col)]
            
        else:
            
            value = 0
            
        fields.append(field)
        
        values.append(value)
        
    return get_recommendations_by_cluster(get_cluster_from_watson_ml(fields, values), values)
		# This function takes a cluster and the quantity of every product already purchased or in the user's cart & returns the data frame of recommendations for the PixieApp
from pyspark.sql.functions import desc

def get_recommendations_by_cluster_app(cluster, purchased_quantities):                                                        
    
    # Existing customer products
    
    customer_products = []
    

    
    
    
    for i in range(0, len(product_cols)):
        
        if purchased_quantities[i] > 0:
            
            customer_products.append((product_cols[i], purchased_quantities[i]))
        
            
    df_customer_products = sc.parallelize(customer_products).toDF(["PRODUCT","COUNT"])
    
    # Get popular products in the cluster
    
    cluster_products = get_popular_products_in_cluster(cluster)
    
    df_cluster_products = sc.parallelize(cluster_products).toDF(["PRODUCT","COUNT"])
    
    df_cluster_products = sc.parallelize(cluster_products).toDF
    
    
    # Filter out products the user has already purchased
    
    
    df_recommended_products = df_cluster_products.alias('cl').join(df_customer_products.alias('cu'), df_cluster_products['PRODUCT'] == df_customer_products['PRODUCT'], 'leftouter')
    

    df_recommended_products = df_recommended_products.filter('cu.PRODUCT IS NULL').select('cl.PRODUCT','cl.COUNT').sort(desc('cl.COUNT'))
    
    return df_recommended_products     
    
# PixieDust sample application

from pixiedust.display.app import *

@PixieApp

class RecommenderPixieApp:
                                                          
    def setup(self):
        
        self.product_cols = product_cols
        
    def computeUserRecs(self, shoppingcart):                 
         
        #format products and quantities from shopping cart display data
    
        lst = list(zip(*[(item.split(":")[0],int(item.split(":")[1])) for item in shoppingcart.split(",")]))
        
        products = list(lst[0])
        
        quantities = list(lst[1])
        
        
        
    
        #format for the Model function
        
        lst = list(zip(*[('sum({})'.format(item),quantities[products.index(item)] if item in products else 0) for item in self.product_cols]))
        
        fields = list(lst[0])
        
        values = list(lst[1])
        
        #call the run Model function
    
        
        recommendations_df = get_recommendations_by_cluster_app(get_cluster_from_watson_ml(fields, values), values)
        
        recs = [row["PRODUCT"] for row in recommendations_df.rdd.collect()]
        
        return recs[:5]
    
    
    @route(shoppingCart="*")
    
    def _recommendation(self, shoppingCart):
        
        recommendation = self.computeUserRecs(shoppingCart)
        
        self._addHTMLTemplateString(
        """
        <table style="width:100%"> {% for item in recommendation %} <tr> <td type="text" style="text-align:left">{{item}}</td> </tr> {% endfor %} </table>
        """, recommendation = recommendation)
        
        
    

        
    @route()
    def main(self):
        return """
        <script>
        function getValuesRec(){
            return $( "input[id^='prod']" )
            .filter(function( index ) {
                return parseInt($(this).val()) > 0;})
            .map(function(i, product) {
                return $(product).attr("name") + ":" + $(product).val();
            }).toArray().join(",");}
            
        function getValuesCart(){
            return $( "input[id^='prod']" )
            .filter(function( index ) {
                return parseInt($(this).val()) > 0; })
            .map(function(i, product) {
                return $(product).attr("name") + ":" + $(product).val();
            }).toArray(); }
        
        function populateCart(field) {
            user_cart = getValuesCart();
            $("#user_cart{{prefix}}").html("");
            if (user_cart.length > 0) {
                for (var i in user_cart) {
                    var item = user_cart[i];
                    var item_arr = item.split(":")
                    $("#user_cart{{prefix}}").append('<tr><td style="text-align:left">'+item_arr[1]+" "+item_arr[0]+"</td></tr>"); } }
            else { $("#user_cart{{prefix}}").append('<tr><td style="text-align:left">'+ "Cart Empty" +"</td></tr>"); } }
        
        function increase_by_one(field) {
            nr = parseInt(document.getElementById(field).value);
            document.getElementById(field).value = nr + 1;
            populateCart(field); }
        
        function decrease_by_one(field) {
            nr = parseInt(document.getElementById(field).value);
            if (nr > 0) { if( (nr - 1) >= 0) { document.getElementById(field).value = nr - 1; } }
            populateCart(field); } 
        </script>
        
        <table> Products: {% for item in this.product_cols %}
            {% if loop.index0 is divisibleby 4 %} <tr> {% endif %}
                <div class="prod-quantity">
                <td class="col-md-3">{{item}}:</td><td><input size="2" id="prod{{loop.index}}{{prefix}}" class="prods" type="text" 
                    style="text-align:center" value="0" name="{{item}}" /></td>
                <td><button onclick="increase_by_one('prod{{loop.index}}{{prefix}}');">+</button></td>
                <td><button onclick="decrease_by_one('prod{{loop.index}}{{prefix}}');">-</button></td>
                </div>
            {% if ((not loop.first) and (loop.index0 % 4 == 3)) or (loop.last) %} </tr> {% endif %}
        {% endfor %} </table>
        
        <div class="row">
            <div class="col-sm-6"> Your Cart: </div>
            <div class="col-sm-6"> Your Recommendations: <button pd_options="shoppingCart=$val(getValuesRec)" pd_target="recs{{prefix}}"> 
                <pd_script type="preRun"> if (getValuesRec()==""){alert("Your cart is empty");return false;} return true;
                </pd_script>Refresh </button> 
            </div>
        </div>
        
        <div class="row">
        <div class="col-sm-3"> <table style="width:100%" id="user_cart{{prefix}}"> </table> </div> <div class="col-sm-3"> </div>
        <div class="col-sm-3" id="recs{{prefix}}" pd_loading_msg="Calling your model in Watson ML"></div> <div class="col-sm-3"> </div>
        </div>
        """
        
    

#run the app
RecommenderPixieApp().run(runInDialog='false')
