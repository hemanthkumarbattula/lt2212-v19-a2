# LT2212 V19 Assignment 2

From Asad Sayeed's statistical NLP course at the University of Gothenburg.

My name: Hemanth Kumar Battula

## Additional instructions
The gendoc.py takes a folder name as input and prints .csv files as output based on the 
arguments used while running.
The simdoc.py takes .csv as input having column zero as index and all the columns represented
by feature names. The rows are vectors. Prints cosine similarities between different classes
to both console and output.txt file

## Results and discussion
-------------------------------------------------------------------------------------------------------------------------------------
| File names                             | ('grain', 'grain')    | ('grain', 'crude')  | ('crude', 'grain')   | ('crude', 'crude')  |
|----------------------------------------|-----------------------|---------------------|----------------------|---------------------|
| cv_outputfile_no_voc_restriction.csv   | 0.13262168793910395   | 0.0776268579319673  | 0.07762685793196729  | 0.16846136029580638 |
--------------------------------------------------------------------------------------------------------------------------------------
| cv_outputfile_restricted_voc_20.csv    | 0.37230896233716304	 | 0.25005894898638004 | 0.25005894898638     |	0.5232922357746077   |
--------------------------------------------------------------------------------------------------------------------------------------
| tfidf_outputfile_restricted_voc_20.csv | 0.3243782386313046	 | 0.16023570818171123 | 0.16023570818171123  |	0.4328152405665922   |
--------------------------------------------------------------------------------------------------------------------------------------
| tfidf_outputfile_no_voc_restriction.csv| 0.05478643002441667	 | 0.021621651047495197| 0.021621651047495197 |	0.056175959905801445 |
--------------------------------------------------------------------------------------------------------------------------------------
| svdT_countvectoroutputfile_100.csv	 | 0.24763523060928347	 | 0.14283904427243252 | 0.14283904427243252  |  0.3040137260225901  |
--------------------------------------------------------------------------------------------------------------------------------------
| svdT_countvectoroutputfile_1000.csv	 | 0.13318861923061648	 | 0.07776683914671656 | 0.07776683914671656  |	0.1688354782793359   |
--------------------------------------------------------------------------------------------------------------------------------------
| svdT_tfidfoutputfile_100.csv		 | 0.12985653358280963	 | 0.05929754450347478 | 0.05929754450347479  |	0.14895135157490838  |
--------------------------------------------------------------------------------------------------------------------------------------
| svdT_tfidfoutputfile_1000.csv	         | 0.0551517195649355	 | 0.021710958699288095| 0.02171095869928809  |	0.05646208224152123  |
--------------------------------------------------------------------------------------------------------------------------------------           

## Vocabulary restriction.

(Write what you chose for the vocabulary restriction for output file
(2), you can give a brief impressionistic justification for why in one
sentence or less.)

### Result table

(Try to use the Markdown table format for this.)

### The hypothesis in your own words

### Discussion of trends in results in light of the hypothesis

## Bonus answers

(Delete if you're not answering the bonus.)
