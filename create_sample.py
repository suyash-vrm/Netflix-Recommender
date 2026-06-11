import os

def create_sample():
    os.makedirs('sample_data', exist_ok=True)
    
    # 1. Read first 50000 lines of combined_data_1.txt
    ratings_path = os.path.join('data', 'combined_data_1.txt')
    sample_ratings_path = os.path.join('sample_data', 'sample_combined_data.txt')
    
    movies_to_keep = set()
    with open(ratings_path, 'r', encoding='utf-8', errors='ignore') as f_in, \
         open(sample_ratings_path, 'w', encoding='utf-8') as f_out:
         
        current_movie = None
        for i, line in enumerate(f_in):
            if i >= 100000: # 100k lines should be around ~2 MB and ~90,000 ratings
                break
            f_out.write(line)
            line = line.strip()
            if line.endswith(':'):
                current_movie = line[:-1]
                movies_to_keep.add(current_movie)

    # 2. Extract corresponding movies from movie_titles.csv
    movies_path = os.path.join('data', 'movie_titles.csv')
    sample_movies_path = os.path.join('sample_data', 'sample_movie_titles.csv')
    
    with open(movies_path, 'r', encoding='iso-8859-1', errors='ignore') as f_in, \
         open(sample_movies_path, 'w', encoding='utf-8') as f_out:
         
        for line in f_in:
            parts = line.split(',')
            if parts[0] in movies_to_keep:
                f_out.write(line)

if __name__ == '__main__':
    create_sample()
    print("Sample dataset created successfully!")
