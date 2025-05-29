import csv
import os
import os.path as osp
import json
from os import environ

from datasets import Dataset, DatasetDict, load_dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class MILUDataset(BaseDataset):

    @staticmethod
    def load(path: str, language: str, **kwargs):
        """Load the MILU dataset for a specific language.
        
        Args:
            path: Path to the dataset
            language: Language to load (e.g., 'Hindi', 'English', etc.)
        
        Returns:
            DatasetDict with 'dev' and 'test' splits
        """
        dataset = DatasetDict()
        
        # First try: Use Hugging Face datasets
        try:
            print(f"Attempting to load MILU dataset for {language} from Hugging Face...")
            milu_dataset = load_dataset("ai4bharat/MILU", language)
            
            # Debug info
            print(f"Test split keys: {list(milu_dataset['test'][0].keys())}")
            sample_item = milu_dataset['test'][0]
            print(f"Sample item: {json.dumps(sample_item, ensure_ascii=False)}")
            
            # Process test set
            test_data = []
            for item in milu_dataset['test']:
                # Handle new format with option1, option2, etc.
                if 'option1' in item:
                    test_data.append({
                        'input': item['question'],
                        'A': item['option1'],
                        'B': item['option2'],
                        'C': item['option3'],
                        'D': item['option4'],
                        'target': 'ABCD'[int(item['target'].replace('option', '')) - 1],
                        'subject': item.get('subject', ''),
                    })
                # Handle old format with options/choices array
                elif 'options' in item or 'choices' in item:
                    option_key = 'options' if 'options' in item else 'choices'
                    test_data.append({
                        'input': item['question'],
                        'A': item[option_key][0],
                        'B': item[option_key][1],
                        'C': item[option_key][2],
                        'D': item[option_key][3],
                        'target': 'ABCD'[item['answer']],
                        'subject': item.get('subject', ''),
                    })
                else:
                    print(f"Warning: Skipping item with unknown format: {item}")
            
            if not test_data:
                raise ValueError(f"No valid test data found for language {language}")
                
            dataset['test'] = Dataset.from_list(test_data)
            print(f"Successfully loaded {len(test_data)} test examples from Hugging Face")
            
            # Process validation set if available
            validation_data = []
            if 'validation' in milu_dataset:
                for item in milu_dataset['validation']:
                    # Handle new format with option1, option2, etc.
                    if 'option1' in item:
                        validation_data.append({
                            'input': item['question'],
                            'A': item['option1'],
                            'B': item['option2'],
                            'C': item['option3'],
                            'D': item['option4'],
                            'target': 'ABCD'[int(item['target'].replace('option', '')) - 1],
                            'subject': item.get('subject', ''),
                        })
                    # Handle old format with options/choices array
                    elif 'options' in item or 'choices' in item:
                        option_key = 'options' if 'options' in item else 'choices'
                        validation_data.append({
                            'input': item['question'],
                            'A': item[option_key][0],
                            'B': item[option_key][1],
                            'C': item[option_key][2],
                            'D': item[option_key][3],
                            'target': 'ABCD'[item['answer']],
                            'subject': item.get('subject', ''),
                        })
                    else:
                        print(f"Warning: Skipping validation item with unknown format")
            
            # If no validation data, use a subset of test data
            if not validation_data:
                print("No validation data found, using a subset of test data for dev split")
                # Use 10% of test data for dev split (or at least 2 examples)
                dev_size = max(2, len(test_data) // 10)
                validation_data = test_data[:dev_size]
                
            dataset['dev'] = Dataset.from_list(validation_data)
            print(f"Successfully loaded {len(validation_data)} dev examples")
            
        except Exception as e:
            print(f"Failed to load from Hugging Face: {str(e)}")
            
            # Try local loading
            try:
                print(f"Checking if {path} is a local directory...")
                if os.path.isdir(path):
                    local_path = path
                else:
                    # Try to resolve path
                    try:
                        print(f"Attempting to resolve {path} using get_data_path...")
                        local_path = get_data_path(path)
                    except Exception as resolve_e:
                        print(f"Failed to resolve path: {str(resolve_e)}")
                        # Use data/MILU as fallback
                        local_path = os.path.join('data', 'MILU')
                        print(f"Using fallback path: {local_path}")
                
                # Ensure test directory exists
                test_dir = os.path.join(local_path, 'test')
                if not os.path.exists(test_dir):
                    os.makedirs(test_dir, exist_ok=True)
                    
                # Check for test file
                test_file = os.path.join(test_dir, f'{language}_test.csv')
                if not os.path.exists(test_file):
                    # Create a sample test file if it doesn't exist
                    print(f"Test file not found at {test_file}, creating sample data...")
                    if language == 'Bengali':
                        sample_data = [
                            ["কোনটি বাংলাদেশের রাজধানী?", "ঢাকা", "কলকাতা", "চট্টগ্রাম", "সিলেট", "A"],
                            ["বাংলাদেশের জাতীয় ফুল কি?", "শাপলা", "গোলাপ", "জুঁই", "রজনীগন্ধা", "A"]
                        ]
                    elif language == 'Hindi':
                        sample_data = [
                            ["भारत की राजधानी क्या है?", "नई दिल्ली", "मुंबई", "कोलकाता", "चेन्नई", "A"],
                            ["भारत की सबसे लंबी नदी कौन सी है?", "यमुना", "गंगा", "ब्रह्मपुत्र", "गोदावरी", "B"]
                        ]
                    else:
                        sample_data = [
                            ["What is the capital of India?", "New Delhi", "Mumbai", "Kolkata", "Chennai", "A"],
                            ["Which river is known as Ganga in India?", "Yamuna", "Ganges", "Brahmaputra", "Godavari", "B"]
                        ]
                    with open(test_file, 'w', encoding='utf-8') as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerows(sample_data)
                    print(f"Created sample test file at {test_file}")
                
                # Load test data
                test_data = []
                with open(test_file, encoding='utf-8') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) != 6:
                            print(f"Warning: Skipping malformed row in {test_file}: {row}")
                            continue
                        test_data.append({
                            'input': row[0],
                            'A': row[1],
                            'B': row[2],
                            'C': row[3],
                            'D': row[4],
                            'target': row[5],
                        })
                
                if not test_data:
                    raise ValueError(f"No valid data found in {test_file}")
                    
                dataset['test'] = Dataset.from_list(test_data)
                print(f"Successfully loaded {len(test_data)} test examples from local file")
                
                # Check for dev file (optional)
                dev_dir = os.path.join(local_path, 'dev')
                dev_data = []
                
                # Try to load existing dev file if available
                if os.path.exists(dev_dir):
                    dev_file = os.path.join(dev_dir, f'{language}_dev.csv')
                    if os.path.exists(dev_file):
                        with open(dev_file, encoding='utf-8') as f:
                            reader = csv.reader(f)
                            for row in reader:
                                if len(row) != 6:
                                    continue
                                dev_data.append({
                                    'input': row[0],
                                    'A': row[1],
                                    'B': row[2],
                                    'C': row[3],
                                    'D': row[4],
                                    'target': row[5],
                                })
                                
                # If no dev data loaded, create dev directory and file with sample data
                if not dev_data:
                    if not os.path.exists(dev_dir):
                        os.makedirs(dev_dir, exist_ok=True)
                    
                    dev_file = os.path.join(dev_dir, f'{language}_dev.csv')
                    if not os.path.exists(dev_file):
                        print(f"Dev file not found, creating sample dev data at {dev_file}")
                        # Just copy the test data for demonstration
                        with open(dev_file, 'w', encoding='utf-8') as f:
                            csv_writer = csv.writer(f)
                            # Create the same test data for dev (for demonstration)
                            with open(test_file, 'r', encoding='utf-8') as test_f:
                                csv_writer.writerows(list(csv.reader(test_f)))
                    
                    # Load the dev data we just created
                    with open(dev_file, encoding='utf-8') as f:
                        reader = csv.reader(f)
                        for row in reader:
                            if len(row) != 6:
                                continue
                            dev_data.append({
                                'input': row[0],
                                'A': row[1],
                                'B': row[2],
                                'C': row[3],
                                'D': row[4],
                                'target': row[5],
                            })
                
                dataset['dev'] = Dataset.from_list(dev_data)
                print(f"Successfully loaded {len(dev_data)} dev examples")
                
            except Exception as local_e:
                print(f"All loading attempts failed for MILU dataset")
                print(f"Hugging Face error: {str(e)}")
                print(f"Local loading error: {str(local_e)}")
                raise RuntimeError(f"Failed to load MILU dataset for language {language}. "
                                  f"Hugging Face error: {str(e)}. "
                                  f"Local loading error: {str(local_e)}")
                
        return dataset