from agents.osint_agent import OSINTAgent 
a = OSINTAgent() 
result = a.execute('Analyze IP 8.8.8.8') 
print('SUMMARY:', str(result.get('summary',''))[:300]) 
