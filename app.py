from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

# Load pre-trained model
model = pickle.load(open('pipeline.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    return render_template('predict.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extract data from the form
            property_type = request.form.get('property_type', None)
            sector = request.form.get('sector', None)
            bedroom = request.form.get('bedroom', None)
            bathroom = request.form.get('bathroom', None)
            balcony = request.form.get('balcony', None)
            age_possession = request.form.get('agePossession', None)
            built_up_area = request.form.get('built_up_area', None)
            servant_room = request.form.get('servant_room', None)
            store_room = request.form.get('store_room', None)
            furnishing_type = request.form.get('furnishing_type', None)
            luxury_category = request.form.get('luxury_category', None)
            floor_category = request.form.get('floor_category', None)

            # Validate and convert numeric inputs
            bedroom = int(bedroom) if bedroom else 0
            bathroom = int(bathroom) if bathroom else 0
            built_up_area = float(built_up_area) if built_up_area else 0.0
            servant_room = int(servant_room) if servant_room else 0
            store_room = int(store_room) if store_room else 0

            # Prepare the input data
            input_data = [[
                property_type, sector, bedroom, bathroom, balcony, age_possession,
                built_up_area, servant_room, store_room, furnishing_type,
                luxury_category, floor_category
            ]]

            columns = [
                'property_type', 'sector', 'bedRoom', 'bathroom', 'balcony',
                'agePossession', 'built_up_area', 'servant room', 'store room',
                'furnishing_type', 'luxury_category', 'floor_category'
            ]
            
            one_df = pd.DataFrame(input_data, columns=columns)
            
            # Make the prediction
            predicted_price = np.expm1(model.predict(one_df))  # Exponentiate to get actual value
            return render_template('predict.html', predicted_price=predicted_price[0])

        except ValueError as ve:
            return render_template('error.html', message=f"Input conversion error: {ve}")
        except KeyError as ke:
            return render_template('error.html', message=f"Missing form key: {ke}")
        except Exception as e:
            return render_template('error.html', message=f"Prediction error: {e}")

    return render_template('predict.html')


##############################################################################################

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')


import io
import base64

@app.route('/analysiscode')
def analysiscode():
    # Load dataset into a DataFrame
    with open('dataframe.pkl', 'rb') as file:
        df = pickle.load(file)
    # Helper function to convert plots to base64
    def plot_to_base64(fig):
        img = io.BytesIO()
        fig.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        encoded = base64.b64encode(img.getvalue()).decode('utf-8')
        img.close()
        return encoded

    # Visualization 1: Average Price by Property Type (Bar Chart)
    fig, ax = plt.subplots(figsize=(10, 6))
    avg_price = df.groupby('property_type')['price'].mean().sort_values()
    avg_price.plot(kind='bar', color='skyblue', ax=ax)
    ax.set_title('Average Price by Property Type')
    ax.set_ylabel('Average Price (in Crores)')
    ax.set_xlabel('Property Type')
    bar_chart = plot_to_base64(fig)
    plt.close(fig)

    # Visualization 2: Distribution of Property Types (Pie Chart)
    fig, ax = plt.subplots(figsize=(8, 8))
    df['property_type'].value_counts().plot.pie(autopct='%1.1f%%', colors=sns.color_palette('pastel'), ax=ax)
    ax.set_title('Distribution of Property Types')
    ax.set_ylabel('')
    pie_chart = plot_to_base64(fig)
    plt.close(fig)

    # Visualization 3: Price Distribution by Property Type (Boxplot)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='property_type', y='price', data=df, palette='Set2', ax=ax)
    ax.set_title('Price Distribution by Property Type')
    ax.set_ylabel('Price (in Crores)')
    ax.set_xlabel('Property Type')
    ax.tick_params(axis='x', rotation=45)
    boxplot = plot_to_base64(fig)
    plt.close(fig)

    # Visualization 4: Built-up Area Distribution (Histogram)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['built_up_area'], bins=20, color='green', kde=True, ax=ax)
    ax.set_title('Distribution of Built-up Area')
    ax.set_xlabel('Built-up Area (sqft)')
    ax.set_ylabel('Frequency')
    histogram = plot_to_base64(fig)
    plt.close(fig)

    # Visualization 5: Price vs Built-up Area (Scatter Plot)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='built_up_area', y='price', hue='property_type', data=df, palette='viridis', ax=ax)
    ax.set_title('Price vs Built-up Area')
    ax.set_xlabel('Built-up Area (sqft)')
    ax.set_ylabel('Price (in Crores)')
    scatter_plot = plot_to_base64(fig)
    plt.close(fig)

    # Visualization 6: Correlation Heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    corr = df[['price', 'built_up_area']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    ax.set_title('Correlation Heatmap')
    heatmap = plot_to_base64(fig)
    plt.close(fig)

    # Visualization 7: Count of Properties by City (Bar Chart)
    fig, ax = plt.subplots(figsize=(10, 6))
    city_counts = df['sector'].value_counts().head(10)
    city_counts.plot(kind='bar', color='orange', ax=ax)
    ax.set_title('Number of Properties by sector')
    ax.set_ylabel('Count')
    ax.set_xlabel('sector')
    city_count_chart = plot_to_base64(fig)
    plt.close(fig)

    # Visualization 8: Average Price by Number of Bedrooms (Line Plot)
    fig, ax = plt.subplots(figsize=(10, 6))
    avg_price_bedrooms = df.groupby('bedRoom')['price'].mean()
    avg_price_bedrooms.plot(kind='line', marker='o', color='purple', ax=ax)
    ax.set_title('Average Price by Number of Bedrooms')
    ax.set_ylabel('Average Price (in Crores)')
    ax.set_xlabel('Number of Bedrooms')
    ax.grid(True)
    avg_price_bedrooms_chart = plot_to_base64(fig)
    plt.close(fig)


    # Visualization 9:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='property_type', y='price', data=df, ax=ax, palette='Set2')
    ax.set_title('Price Distribution by Property Type')
    ax.set_xlabel('Property Type')
    ax.set_ylabel('Price')
    price_property_type_boxplot = plot_to_base64(fig)
    plt.close(fig)


    # Visualization 10:
    fig, ax = plt.subplots(figsize=(10, 6))
    avg_price_by_sector = df.groupby('sector')['price'].mean().sort_values()
    avg_price_by_sector.plot(kind='bar', color='skyblue', ax=ax)
    ax.set_title('Average Price by Sector')
    ax.set_xlabel('Sector')
    ax.set_ylabel('Average Price')
    avg_price_by_sector_chart = plot_to_base64(fig)
    plt.close(fig)

    # Visualization 11:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='built_up_area', y='price', data=df, ax=ax, color='green')
    ax.set_title('Price vs. Built-up Area')
    ax.set_xlabel('Built-up Area (sqft)')
    ax.set_ylabel('Price')
    scatter_plot_builtup_area = plot_to_base64(fig)
    plt.close(fig)

    # Visualization 12:
    fig, ax = plt.subplots(figsize=(10, 6))
    bedroom_counts = df['bedRoom'].value_counts().sort_index()
    bedroom_counts.plot(kind='bar', color='orange', ax=ax)
    ax.set_title('Number of Properties by Bedroom Count')
    ax.set_xlabel('Number of Bedrooms')
    ax.set_ylabel('Count')
    bedroom_count_chart = plot_to_base64(fig)
    plt.close(fig)

    # Visualization 13:
    fig, ax = plt.subplots(figsize=(10, 6))
    corr = df[['price', 'built_up_area', 'bedRoom', 'bathroom']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    ax.set_title('Correlation Heatmap')
    heatmap_corr = plot_to_base64(fig)
    plt.close(fig)

    # Visualization 14:
    fig, ax = plt.subplots(figsize=(8, 8))
    df['furnishing_type'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax, colors=sns.color_palette('pastel'))
    ax.set_title('Distribution of Furnishing Type')
    furnishing_type_pie_chart = plot_to_base64(fig)
    plt.close(fig)

    # Visualization 15:



    

    # Render Template with Embedded Images
    return render_template(
        'analytics.html',
        bar_chart=bar_chart,
        pie_chart=pie_chart,
        boxplot=boxplot,
        histogram=histogram,
        scatter_plot=scatter_plot,
        heatmap=heatmap,
        city_count_chart=city_count_chart,
        avg_price_bedrooms_chart=avg_price_bedrooms_chart,
        price_property_type_boxplot=price_property_type_boxplot,
        avg_price_by_sector_chart=avg_price_by_sector_chart,
        scatter_plot_builtup_area=scatter_plot_builtup_area,
        bedroom_count_chart=bedroom_count_chart,
        heatmap_corr=heatmap_corr,
        furnishing_type_pie_chart=furnishing_type_pie_chart

    )








@app.route('/recommend')
def recommend():
    return render_template('recommend.html')




with open('df.pkl', 'rb') as file:
    new_df = pickle.load(file)

cosine_sim = pickle.load(open('cosine_sim.pkl', 'rb'))

def top_fac_reco(property_name, similarity=cosine_sim):
    # Find the index of the property in the DataFrame
    try:
        idx = new_df[new_df['PropertyName'] == property_name].index.tolist()[0]
    except IndexError:
        return None  # Return None if the property name is not found

    # Get the cosine similarity scores for the given property
    sim_scores = list(enumerate(similarity[idx]))
    
    # Sort the similarity scores in descending order and get the top 6 (excluding the input property)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Extract the indices of the top 6 similar properties
    sim_scores = sim_scores[1:6]
    
    recommendations = []
    for i in sim_scores:
        recommended_property = new_df['PropertyName'].iloc[i[0]]
        recommendations.append(recommended_property)

    return recommendations



@app.route('/recommend_houses', methods=['POST'])
def recommend_houses():
    user_input = request.form['user_input']
    
    # Call the recommendation function
    recommendations = top_fac_reco(user_input)
    
    if recommendations is None:
        recommendations = ['No property found with that name. Please try again.']
    
    # Render the recommendations page with the recommendations list
    return render_template('recommend.html', recommendations=recommendations)





if __name__ == "__main__":
    app.run(debug=True)
