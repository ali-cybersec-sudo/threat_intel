from dotenv import load_dotenv 
import os 
load_dotenv() 
key = os.getenv('OPENROUTER_API_KEY','') 
print('KEY FOUND:', bool(key)) 
print('LENGTH:', len(key)) 
