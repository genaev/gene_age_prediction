# Gene age prediction
The program accepts alignment in the blast6out format as an input. Predicts age for each gene. Returns, depending on the predicted age, N best hits.

#### Usage: 
```
usage: gene_age_prediction.py [-h] [-i INPUT] [-s SUPPLEMENTARY] [-o OUTPUT]
                              [--thr_other THR_OTHER] [--thr_young THR_YOUNG]
                              [infile] [errfile] [outfile]

positional arguments:
  infile                input alignment blast6out file
  errfile               output log file
  outfile               output alignment blast6out file

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        input filename
  -s SUPPLEMENTARY, --supplementary SUPPLEMENTARY
                        output supplementary file
  -o OUTPUT, --output OUTPUT
                        output filename
  --thr_other THR_OTHER
                        threshold for other age groups, default=10
  --thr_young THR_YOUNG
                        threshold for young age group, default=5
  --n_jobs
                        jobs number, default=1
```

#### Example:
`./python gene_age_prediction.py -i test.csv -s supliment.csv -o alignment_filtred.blast6out`

`blastp -query query.fasta -db target_idx.fasta -outfmt 6 | python3 gene_age_pred.py --n_jobs 2 --thr_other 5 --thr_young 2 > aling.csv` - with stdin, stderr and stdout application.

`python3 gene_age_pred.py < test.csv 1> aling.csv 2> suplim.csv` 

`./python gene_age_prediction.py -i test.csv -s supliment.csv -o alignment_filtred.blast6out --thr_other 7 --thr_young 2  --n_jobs 4` - full calling line
