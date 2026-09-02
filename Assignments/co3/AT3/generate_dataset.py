import pandas as pd

topics = [
    ("Deep Learning Fundamentals", "Learn neural networks, backpropagation, optimization, CNNs and deep learning concepts.", "Deep Learning", "Beginner"),
    ("Advanced Deep Learning", "Study advanced neural network architectures, representation learning and optimization techniques.", "Deep Learning", "Advanced"),
    ("PyTorch for Beginners", "Learn PyTorch tensors, datasets, neural networks, training loops and model evaluation.", "PyTorch", "Beginner"),
    ("PyTorch Deep Learning", "Build deep learning models using PyTorch including CNNs, RNNs and transfer learning.", "PyTorch", "Intermediate"),
    ("Computer Vision with PyTorch", "Develop image classification and object detection systems using PyTorch.", "Computer Vision", "Intermediate"),
    ("Natural Language Processing", "Learn text processing, embeddings, classification, sentiment analysis and NLP models.", "NLP", "Intermediate"),
    ("Machine Learning Fundamentals", "Learn supervised learning, unsupervised learning, regression, classification and clustering.", "Machine Learning", "Beginner"),
    ("TensorFlow Fundamentals", "Build and train neural networks using TensorFlow and Keras.", "TensorFlow", "Beginner"),
    ("Python Programming", "Learn Python programming, variables, functions, classes, modules and file handling.", "Programming", "Beginner"),
    ("Data Science with Python", "Learn Python, pandas, NumPy, visualization and machine learning for data science.", "Data Science", "Intermediate"),
    ("Artificial Intelligence", "Explore artificial intelligence, intelligent agents, search algorithms and machine learning.", "Artificial Intelligence", "Beginner"),
    ("Reinforcement Learning", "Learn agents, rewards, policies, value functions and reinforcement learning algorithms.", "AI", "Advanced"),
    ("Generative AI", "Learn generative artificial intelligence, language models, prompting and AI applications.", "Generative AI", "Intermediate"),
    ("Large Language Models", "Understand transformers, language models, embeddings and large language model applications.", "Generative AI", "Advanced"),
    ("Transformers and NLP", "Learn transformer architectures, attention mechanisms and modern NLP applications.", "NLP", "Advanced"),
    ("Data Analytics", "Learn data analysis, visualization, statistics and practical analytical techniques.", "Data Analytics", "Beginner"),
    ("SQL for Data Science", "Learn SQL queries, joins, aggregation, filtering and database analysis.", "Database", "Beginner"),
    ("Cloud Computing", "Learn cloud platforms, virtualization, storage, networking and cloud application development.", "Cloud", "Intermediate"),
    ("Computer Networks", "Understand networking protocols, TCP/IP, routing, switching and network security.", "Networking", "Intermediate"),
    ("Cyber Security Fundamentals", "Learn authentication, encryption, vulnerabilities, security principles and cyber defense.", "Cyber Security", "Beginner"),
    ("Statistics for Machine Learning", "Learn probability, statistics, distributions and statistical methods used in machine learning.", "Statistics", "Intermediate"),
    ("MLOps Fundamentals", "Learn machine learning deployment, model monitoring, pipelines and production workflows.", "MLOps", "Advanced"),
    ("AI Ethics", "Explore responsible AI, fairness, transparency, privacy and ethical machine learning.", "AI Ethics", "Beginner"),
    ("Time Series Forecasting", "Learn forecasting, trends, seasonality and predictive models for time series data.", "Data Science", "Intermediate"),
    ("Recommender Systems", "Build recommendation systems using collaborative filtering, content-based methods and embeddings.", "Machine Learning", "Advanced"),
]

rows = []

course_id = 1

# Generate 220 records by creating variations
for repeat in range(9):
    for title, description, category, level in topics:
        rows.append({
            "course_id": course_id,
            "title": title if repeat == 0 else f"{title} - Part {repeat + 1}",
            "description": description,
            "category": category,
            "level": level
        })
        course_id += 1

# Keep exactly 220 records
rows = rows[:220]

df = pd.DataFrame(rows)

df.to_csv("Assignment/co3/AT3/dataset/courses.csv", index=False)

print("Dataset created successfully!")
print("Number of courses:", len(df))
print("\nCategories:")
print(df["category"].value_counts())

print("\nFirst 5 records:")
print(df.head())