import streamlit as st
import pandas as pd 
import streamlit.components.v1 as components
Orders = { "order_id" : [1,2,3,4,5], 
          "customer_id": [259,1212,523,175,1324], 
          "order_status": [4,4,4,4,4], 
          "order_date": ["2016-01-01", " 2016-01-01"," 2016-01-01"," 2016-01-03"," 2016-01-03"], 
          "required_date": [ "2016-01-03","2016-01-04"," 2016-01-05"," 2016-01-04"," 2016-01-06"], 
          "shipped_date": ["2016-01-03","2016-01-03","2016-01-03","2016-01-05","2016-01-06"], 
          "store_id": [1,2,2,1,2], 
          "staff_id" : [2,6,7,3,6],
          }

Orders = pd.DataFrame(Orders)

customers ={
 "customer_id" : [1,2,3,4,5] ,
 "first_name" : ["Debra","Kasha","Tameka","Daryl" ,"Charolette"],
 "last_name" : ["Burks","Todd","Fisher","Spence","Rice"] ,
 "phone " : ["(916) 381-6003","(916) 381-6003","(916) 381-6003"," (916) 381-6003","(916) 381-6003"], 
 "email": ["debra.burks@yahoo.com","kasha.todd@yahoo.com","tameka.fisher@aol.com","daryl.spence@aol.com","charolette.rice@msn.com"],
 "street " : ["9273 Thorne Ave.","910 Vine Street","769C Honey Creek St.","988 Pearl Lane","107 River Dr."],
 "city  " : ["Orchard Park",  "Campbell", "Redondo Beach", "Uniondale","Sacramento"] ,
 "state" : ["NY","CA","CA","NY","CA"] ,
 "zip_code" : ["14127","95008","90278","11553"," 95820"]
}
customers = pd.DataFrame(customers)

order_items = {
     "order_id": [1, 1, 1, 1, 1] ,
     "item_id": [1, 2, 3, 4, 5],
     "product_id": [20, 8, 10, 16, 4] ,
     "quantity": [1, 2, 2, 2, 1],
     "list_price": [599.99, 1799.99, 1549.0, 599.99, 2899.99] ,
     "discount": [0.2, 0.07, 0.05, 0.05, 0.2]

}
order_items = pd.DataFrame(order_items)

products ={
     "product_id": [1, 2, 3, 4, 5],
     "product_name": ['Trek 820 - 2016', 'Ritchey Timberwolf Frameset - 2016', 'Surly Wednesday Frameset - 2016', 'Trek Fuel EX 8 29 - 2016', 'Heller Shagamaw Frame - 2016'],
     "brand_id": [9, 5, 8, 9, 3],
     "category_id": [6, 6, 6, 6, 6],
     "model_year": [2016, 2016, 2016, 2016, 2016],
     "list_price":[379.99, 749.99, 999.99, 2899.99, 1320.99]
} 
products  = pd.DataFrame(products )
categories ={
    "category_id": [1, 2, 3, 4, 5] ,
    "category_name": ['Children Bicycles', 'Comfort Bicycles', 'Cruisers Bicycles', 'Cyclocross Bicycles', 'Electric Bikes']
}
categories = pd.DataFrame(categories)
stores = {
     "store_id": [1, 2, 3, 4, 5] ,
     "store_name":  ['Santa Cruz Bikes', 'Baldwin Bikes', 'Rowlett Bikes', 'Santa Cruz Bikes', 'Baldwin Bikes'],
     "phone"   : ['(831) 476-4321', '(516) 379-8888', '(972) 530-5555', '(831) 476-4321', '(516) 379-8888'],
     "email"  : ['santacruz@bikes.shop', 'baldwin@bikes.shop', 'rowlett@bikes.shop', 'santacruz@bikes.shop', 'baldwin@bikes.shop'],
     "street" : ['3700 Portola Drive', '4200 Chestnut Lane', '8000 Fairway Avenue', '3700 Portola Drive', '4200 Chestnut Lane'],
     "city"  : ['Santa Cruz', 'Baldwin', 'Rowlett', 'Santa Cruz', 'Baldwin'],
     "state" :['CA', 'NY', 'TX', 'CA', 'NY'],
     "zip_code": ['95060', '11432', '75088', '95060', '11432']
}
stores= pd.DataFrame(stores)
staffs ={
     "staff_id":[1, 2, 3, 4, 5],
     "first_name":['Fabiola', 'Mireya', 'Genna', 'Virgie', 'Jannette'] ,
     "last_name":['Jackson', 'Copeland', 'Serrano', 'Wiggins', 'David'] ,
     "email":['fabiola.jackson@bikes.shop', 'mireya.copeland@bikes.shop', 'genna.serrano@bikes.shop', 'virgie.wiggins@bikes.shop', 'jannette.david@bikes.shop'] ,
     "phone":['(831) 555-5554', '(831) 555-5555', '(831) 555-5556', '(831) 555-5557', '(516) 379-4444'],
     "active":[1, 1, 1, 1, 1] ,
     "store_id":[1, 1, 1, 1, 2] ,
     "manager_id": [0, 1, 2, 2, 1]
}
staffs = pd.DataFrame(staffs )

st.header("Bike Store Sales Dashboard: SQL & Tableau Unleash Data-Driven Success", divider='blue')
st.subheader("**Project Overview**")
st.write(""" 
\t The Bike Store Sales Dashboard project involved the creation of a powerful tool to enhance a local bike store's operational efficiency and decision-making. Beginning with data normalization in MySQL, the project addressed performance concerns through denormalization, optimizing query speed and efficiency. The denormalized database featured key columns like order_id, Customer_Name, and Revenue. Integrated seamlessly into Tableau, the dashboard offered dynamic visualization and comprehensive functionalities. Users could explore summarized information, yearly trends, monthly revenue, and granular details by state, store, bike category, and sales representative. The user-focused design allowed for customization, and the dashboard facilitated in-depth revenue analysis, identifying top customers for targeted strategies. Overall, the project transformed raw sales data into actionable insights, empowering the bike store to make informed decisions and optimize business operations.
""")

st.subheader("**Problem Statement**")

st.write("The bike store faced critical challenges in effectively monitoring and leveraging its sales data to drive informed business decisions. The existing data management system, relying on normalized data in MySQL, struggled with query performance, hindering the store's ability to quickly extract valuable insights. Recognizing the need for a solution that not only addressed these performance issues but also provided a user-friendly interface for comprehensive sales analysis, the project aimed to denormalize the data and integrate it seamlessly into Tableau. The overarching problem was the lack of an efficient, visually intuitive tool to transform raw sales data into actionable insights, limiting the store's capacity to optimize operations, identify trends, and tailor strategies to enhance overall performance.")

st.subheader("__**Data Evolution**__ ")
st.subheader("__ ** From Normalized to Denormalized Form**__")
st.write("""*Note:*
*For a seamless demonstration of the transformation from normalized to denormalized data, I utilized Python's Pandas library. This streamlined approach not only simplifies the showcase but also minimizes potential latency by consolidating data into Pandas DataFrames. The result is a faster and more responsive user experience.*""")
st.write("This table corresponds to the Orders dataset, capturing key details including order ID, customer ID, status, dates, store ID, and staff ID. Each entry provides specific information about individual orders, offering insights into customer transactions and order processing.")
st.dataframe(Orders)

st.write("This table contains customer details such as customer ID, first name, last name, phone number, email, street address, city, state, and zip code. Each entry in the table represents a unique customer, providing comprehensive information about their contact details and residence.")
st.dataframe(customers)

st.write("The database includes key tables crucial for bike store management. The **order items** table details items within orders, **products** covers product specifics, and **categories** defines product categories. The **stores** table provides store details, and **staffs** contains essential staff information. Each table plays a vital role in efficiently managing inventory, sales, and staff data for the bike store.")
st.write("**Denormalized Data**")
st.write("SQL code creates a database named **bikes_store** and a table named **denormalised_bike_data**. It then populates this table by joining various tables, calculating total units and revenue, and grouping the data by order attributes. This denormalized view consolidates information from multiple tables for efficient analysis and reporting in the bike store context.")

bike_store_data = {
     "order_id":[1,1,1,1,1],
     "customer_Name": ["Johnathan Velazquez","Johnathan Velazquez","Johnathan Velazquez","Johnathan Velazquez","Johnathan Velazquez"],	
     "city":['Pleasanton', 'Pleasanton', 'Pleasanton', 'Pleasanton', 'Pleasanton'],
     "state":['CA', 'CA', 'CA', 'CA', 'CA'],	
     "order_date":['2016-01-01', '2016-01-01', '2016-01-01', '2016-01-01', '2016-01-01'],	
     "total_Units":[4, 2, 4, 2, 4],	
     "revenue":[2400, 1200, 6196, 5800, 7200],	
     "category_name":['Electra Townie Original 7D EQ - 2016', "Electra Townie Original 7D EQ - Women's - 2016", 'Surly Straggler - 2016', 'Trek Fuel EX 8 29 - 2016', 'Trek Remedy 29 Carbon Frameset - 2016'],	
     "store_name":['Santa Cruz Bikes', 'Santa Cruz Bikes', 'Santa Cruz Bikes', 'Santa Cruz Bikes', 'Santa Cruz Bikes'],	
     "sales_rep":['Mireya Copeland', 'Mireya Copeland', 'Mireya Copeland', 'Mireya Copeland', 'Mireya Copeland'],	
     "product_name":['Cruisers Bicycles', 'Cruisers Bicycles', 'Cyclocross Bicycles', 'Mountain Bikes', 'Mountain Bikes'],
}

code ="""
create database if not exists bikes_store;
use bikes_store;
CREATE TABLE if not exists denormalised_bike_data (
    order_id DECIMAL,
    Customer_Name VARCHAR(300),
    city VARCHAR(300),
    state VARCHAR(20),
    order_date DATE,
    Total_Units INT,
    Revenue DECIMAL,
    category_name VARCHAR(600),
    store_name VARCHAR(600),
    sales_rep VARCHAR(600),
    product_name VARCHAR(900)
);
truncate denormalised_bike_data; 
insert into denormalised_bike_data select
    ord.order_id,
    concat(cus.first_name, " ", cus.last_name) as "Customer Name",
    cus.city,
    cus.state,
    ord.order_date,
    sum(ite.quantity) as "Total Units",
    sum(ite.quantity * ite.list_price) as "Revenue",
    pro.product_name,
    cat.category_name,
    sto.store_name,
    concat (sta.first_name, " ", sta.last_name) as "sales_rep"
from sales.orders ord join
sales.customers cus
ON ord.customer_id = cus.customer_id 
join sales.order_items ite 
on ord.order_id = ite.order_id
join production.products  pro
on ite.product_id = pro.product_id
join production.categories cat
on pro.category_id = cat.category_id
join sales.stores sto 
on sto.store_id = ord.store_id
join sales.staffs sta 
on ord.staff_id = sta. staff_id
group by
    ord.order_id,
    concat(cus.first_name, " ", cus.last_name),
    cus.city,
    cus.state,
    ord.order_date,
    pro.product_name, 
    cat.category_name,
    sto.store_name,
    concat (sta.first_name, " ", sta.last_name)
order by ord.order_id;
"""
st.code(code, language ="sql")

st.write("""
         The denormalized data, represented by the **denormalised_bike_data** table, amalgamates information for enhanced readability and reduced complexity. The table includes details such as order ID, customer name, city, state, order date, total units, revenue, category name, store name, sales representative, and product name. Each entry provides a consolidated view of the order, customer, and product-related information, streamlining data representation for improved comprehension and simplified analysis.
         """)
st.dataframe(bike_store_data)

st.subheader("Solution")

st.write("""
          The sales dashboard served as a transformative solution to address the identified challenges in managing and analyzing sales data efficiently. By implementing a denormalization process in the MySQL database, the dashboard significantly improved query performance, ensuring faster and more responsive data retrieval. This strategic choice allowed for streamlined and simplified queries, overcoming the complexities associated with normalized data structures. The technologies employed included MySQL for data storage, Pandas for data transformation, and Tableau for dynamic and interactive visualizations. Leveraging Tableau's capabilities, the dashboard provided a user-friendly interface, empowering users to explore and analyze sales data effortlessly. Notable features included comprehensive data summaries for the entire dataset, dynamic yearly summaries for trend analysis, and interactive visualizations that facilitated a more in-depth understanding of the sales landscape. Overall, the integrated use of these technologies and tools not only addressed the initial challenges but also positioned the sales dashboard as a powerful tool for data-driven decision-making and optimization of sales strategies in the bike store.

""")


components.html("""<div class='tableauPlaceholder' id='viz1705368797432' style='position: relative'><noscript><a href='#'><img alt='Dashboard 1 ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Br&#47;BridgingInsightsNavigatingBikeStoreOperationsthroughDenormalizedDataandExecutiveDashboards&#47;Dashboard1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='path' value='views&#47;BridgingInsightsNavigatingBikeStoreOperationsthroughDenormalizedDataandExecutiveDashboards&#47;Dashboard1?:language=en-US&amp;:embed=true&amp;publish=yes' /> <param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Br&#47;BridgingInsightsNavigatingBikeStoreOperationsthroughDenormalizedDataandExecutiveDashboards&#47;Dashboard1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1705368797432');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='800px';vizElement.style.height='1627px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='800px';vizElement.style.height='1627px';} else { vizElement.style.width='100%';vizElement.style.height='2577px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>""",height=1700,width=800)


st.header("""Navigating Database Transitions \n Converting PostgreSQL to MySQL - Challenges Encountered and Solutions Explored""")

st.write("""Facing diverse challenges, my journey involved overcoming five significant obstacles. Initially, I took on the task of converting data from PostgreSQL to MySQL, achieving success with a custom Python script utilizing file and string operations. The second challenge emerged when I grappled with data scattered across different databases, prompting meticulous efforts to consolidate it. Addressing the third obstacle required handling data type disparities in the original dataset, leading to a conversion to the compatible decimal type in MySQL. Building on a structured foundation, I personally established a dedicated database and a normalized table, streamlining integration with Tableau. As my journey unfolded, the final hurdle presented itself – the need for MySQL Connector. This demanded resourceful research and a download phase, ensuring a successful connection. The narrative concludes with a personal sense of accomplishment, having navigated through these challenges and successfully connected the dots in the realm of databases and data visualization tools.""")


st.subheader("Conclusion")
st.write(""" 
The Bike Store Sales Dashboard project effectively addressed challenges faced by a local bike store, enhancing sales data utilization for informed decision-making. The shift from a normalized MySQL structure to a denormalized form not only optimized query performance but also seamlessly integrated with Tableau for dynamic visualizations. This solution resolved the lack of an efficient tool for transforming raw sales data into actionable insights, with the denormalized data in the denormalised_bike_data table consolidating information for improved readability and reduced complexity.

The data evolution, demonstrated using Python's Pandas library, showcased a streamlined approach, ensuring a faster user experience. The SQL code facilitated the creation of a denormalized database, populating it with joined data and computing key metrics. The resulting sales dashboard, leveraging MySQL, Pandas, and Tableau, significantly improved query performance and provided a user-friendly interface for comprehensive sales analysis, featuring dynamic yearly trends and interactive visualizations.

As part of the same project, another endeavor involved converting data from PostgreSQL to MySQL, successfully overcoming challenges such as scattered data and data type disparities. The journey concluded with a sense of accomplishment, having navigated through obstacles in the realms of databases and data visualization tools. In summary, both projects underscored the transformative impact of data-driven solutions, empowering businesses to make informed decisions and optimize their operations.""")


st.subheader("Appendix")
st.write("""http://tinyurl.com/linktodatalink""")
