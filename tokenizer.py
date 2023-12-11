from datasets import load_dataset 
import numpy as np 
import os 
save_path = "data/wikitext.txt"

def save_dataset_to_text(file_path):
    # Load the dataset
    dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", split="train")

    # Check if the file already exists
    if not file_path.endswith('.txt'):
        file_path += '.txt'
    
    if not os.path.exists(file_path):
        # Save each example in the dataset to the text file
        with open(file_path, 'w', encoding='utf-8') as file:
            for words in dataset['text']:
                file.write(words + '\n')
        print(f"Dataset saved to '{file_path}'.")
    elif os.path.getsize(file_path) < 1:
        print(f"The file '{file_path}' already exists but is empty.")
        with open(file_path, 'w', encoding='utf-8') as file:
            for words in dataset['text']:
                
                file.write(words + '\n')
                print(f"Dataset saved to '{file_path}'.")
    else: 
        print(f"The file '{file_path}' already exists. Dataset not saved.")


# Example usage:

text_file_path = 'data/wikitext_dataset.txt'

save_dataset_to_text(text_file_path)

def read_file_to_array(file_path):
    lines_array = []

    try:
        with open(file_path, 'r') as file:
            for line in file:
                cleaned_line = line.strip()
                
                if cleaned_line:
                    lines_array.append(cleaned_line)

        return lines_array

    except FileNotFoundError:
        print(f"The file '{file_path}' does not exist.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
result_array = read_file_to_array(text_file_path)

from transformers import pipeline
sentiment_classifier = pipeline("sentiment-analysis")
sentiments_by_line = np.array([])
test = sentiment_classifier(result_array[0])
for text in result_array[:50]:
    result = sentiment_classifier(text)
    sentiments_by_line = np.append(sentiments_by_line, result)
    
print("Sentiments By Line \n")
print(sentiments_by_line)

text = result_array[:50]
new_text = []
string = ''
index_of_text = 1
sentiment_index = 0
index_of_new_text = 0
current_sentiment = sentiments_by_line[0]['label']
for line in text:
    new_sentiment = sentiments_by_line[sentiment_index]['label']
    string = string + line
    if current_sentiment != new_sentiment:
        string = [string]
        new_text = new_text + string
        current_sentiment = new_sentiment
        string = ''
        index_of_new_text = index_of_new_text + 1
    index_of_text = index_of_text + 1
    sentiment_index = sentiment_index + 1

print(len(new_text))

def split_array(input_array, chunk_size):
    """
    Split an array into chunks of a specified size.
    
    Parameters:
    - input_array: The input array to be split.
    - chunk_size: The size of each chunk.
    
    Returns:
    A list of arrays, each containing up to chunk_size elements.
    """
    return [input_array[i:i + chunk_size] for i in range(0, len(input_array), chunk_size)]
        
# comparing Ner text up to 50 fed at once ves looping tokens 
array_17 = split_array(result_array, 17)

ner = pipeline("ner", grouped_entities=True)
ner_full_results = ner(result_array[:50])
print("NER full results completed")
ner_token_results = []
ner_17_results = []
for text in new_text:
    ner_token_results = ner_token_results + ner(text)
print("NER token results completed")

result1 = ner(array_17[0])
print("NER 17-1 results completed")
result2 = ner(array_17[1])
print("NER 17-2 results completed")
result3 = ner(array_17[2])
print("NER 17-3 results completed")
result4 = ner(array_17[3])
print("NER 17-4 results completed")
result5 = ner(array_17[4])
print("NER 17-5 results completed")
result6 = ner(array_17[5])
print("NER 17-6 results completed")
result7 = ner(array_17[6])
print("NER 17-7 results completed")
result8 = ner(array_17[7])
print("NER 17-8 results completed")
result9 = ner(array_17[8])
print("NER 17-9 results completed")
result10 = ner(array_17[9])
print("NER 17-10 results completed")
result11 = ner(array_17[10])
print("NER 17-11 results completed")
result12 = ner(array_17[11])
print("NER 17-12 results completed")
result13 = ner(array_17[12])
print("NER 17-13 results completed")
result14 = ner(array_17[13])
print("NER 17-14 results completed")
result15 = ner(array_17[14])
print("NER 17-15 results completed")
result16 = ner(array_17[15])
print("NER 17-16 results completed")
result17 = ner(array_17[16])
print("NER 17-17 results completed")
result17 = ner(array_17[17])
print("NER 17-18 results completed")
ner_17_results = result1+ result2 + result3 + result4 + result5 + result6 + result7 + result8 + result9 + result10 + result11 + result12 + result13 + result14 + result15 + result16 + result17
ner_17_results = [line for line in ner_17_results if line] 
print("NER 17 results all completed")

print(ner_full_results)
print("\n")
print(ner_token_results)
print("\n")
print(ner_17_results)
 
filename = 'results/semantic-tokeniser-ner.txt'
with open(filename, 'w', encoding='utf-8') as file:
    file.write("NER Full Array Results [:50] \n")
    for entry in ner_full_results:
        file.write(str(entry) + '\n')
    file.write("NER Token Results [:50] \n")
    for entry in ner_token_results:
        file.write(str(entry) + '\n')
    file.write("NER 17 Array Split Results [:50] \n")
    for entry in ner_17_results:
        file.write(str(entry) + '\n')
            
print("Data written")
        
print("Relevant info\n")
print("Array length\n")
print(len(ner_full_results))
print(len(ner_token_results))
print(len(ner_17_results))

def calculations(data):
    misc_score = 0 
    misc_count = 0 
    loc_score = 0
    loc_count = 0
    per_score = 0
    per_count = 0
    org_score = 0
    org_count = 0

    # Iterate over each list of arrays
    for array_list in data:
        # Iterate over each dictionary in the array
   
        if isinstance(array_list, list):
            for entry in array_list:
                # Extract entity group and score from the dictionary
                entity_group = entry['entity_group']
                # print(entity_group)
                score = entry['score']
                # print(score)
                # Update counters and sums based on the entity group
                if entity_group == 'MISC':
                    misc_score += score
                    misc_count += 1
                elif entity_group == 'LOC':
                    loc_score += score
                    loc_count += 1
                elif entity_group == 'PER':
                    per_score += score
                    per_count += 1
                elif entity_group == 'ORG':
                    org_score += score
                    org_count += 1

        # Calculate averages
    misc_avg = misc_score/misc_count
    loc_avg = loc_score/loc_count
    per_avg = per_score/per_count
    org_avg = org_score/org_count

        # Print the results
    print("MISC: Count =", misc_count, "Average Score =", misc_avg)
    print("LOC: Count =", loc_count, "Average Score =", loc_avg)
    print("PER: Count =", per_count,  "Average Score =", per_avg)
    print("ORG: Count =", org_count,  "Average Score =", org_avg)

    

from collections import defaultdict

def calc_dict(data):
    category_counts = defaultdict(int)
    category_scores = defaultdict(float)
    misc_score = 0 
    misc_count = 0 
    loc_score = 0
    loc_count = 0
    per_score = 0
    per_count = 0
    org_score = 0
    org_count = 0
    for entry in data:
        entity_group = entry['entity_group']
        score = entry['score']
        
        if entity_group == 'MISC':
            misc_score += score
            misc_count += 1
        elif entity_group == 'LOC':
            loc_score += score
            loc_count += 1
        elif entity_group == 'PER':
            per_score += score
            per_count += 1
        elif entity_group == 'ORG':
            org_score += score
            org_count += 1
    
    
    misc_avg = misc_score/misc_count
    loc_avg = loc_score/loc_count
    per_avg = per_score/per_count
    org_avg = org_score/org_count
    # Update counters and sum of scores
    print("MISC: Count =", misc_count, "Average Score =", misc_avg)
    print("LOC: Count =", loc_count, "Average Score =", loc_avg)
    print("PER: Count =", per_count,  "Average Score =", per_avg)
    print("ORG: Count =", org_count,  "Average Score =", org_avg)
 
print("NER Token Analysis \n")
calc_dict(ner_token_results)
print("\n")
print("NER Full Analysis \n")
calculations(ner_full_results)
print("\n")
print("NER 17 Analysis \n")
calculations(ner_17_results)





# sentiments_6n_1 = np.array([])

# for text in dataset['text'][:50]:
#      text_2_classify = np.append(text_2_classify, text)   
    
# print(type(text_2_classify[1]))


# for text in text_2_classify:
#     np.append(sentiments_text, classifier(sentiments_text))

# text_6n_1 = np.array([])


        
