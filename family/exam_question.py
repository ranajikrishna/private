
from my_quant_library import * 
from ast import literal_eval

def main(argv = None):
	
	# Input filename.
	inputFile = '/Users/vashishtha/myGitCode/family/exam_question_historical_data.xlsx'      
	
	data = pd.ExcelFile(inputFile).parse()	# Ticker names.
	pdb.set_trace()
	
	data['6 marks chapters']= data['6 marks chapters'].apply(literal_eval)	

	pdb.set_trace()	

	return(0)


if __name__ == '__main__':
	status = main()
	sys.exit(status)
