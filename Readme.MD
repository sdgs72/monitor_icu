<h1> Instructions </h1>

<h2> Original paper </h2>

<code> Yu, K., Zhang, M., Cui, T., & Hauskrecht, M. (2020). Monitoring ICU Mortality Risk with A Long Short-Term Memory Recurrent Neural Network. Pacific Symposium on Biocomputing. Pacific Symposium on Biocomputing, 25, 103–114. </code>


<h2> Prerequirements </h2>
<hr>
<ul>
    <li> MySQL v8.1++ </li>
    <li> Python 3.9 and pip3 </li>
    <li> OS that can run sh scripts</li>
    <li> DBViewer (recommended for easy SQL to CSV imports) </li>
    <li> run the following to install all python3 libraries required <code> pip3 install -r requirements.txt </code>
</ul>


<h2> Getting the datasource </h2>
<hr>
<ol>
    <li> Get the full mimic3 dataset from https://physionet.org/content/mimiciii/1.4/ extract all the CSVs file to a temporary directory</li>
    <li> Clone/Download the mimic-code repository used to build tables in for the raw mimic3 data source https://github.com/MIT-LCP/mimic-code </li>
    <li> Follow the instructions on how to build and import the raw mimic3 csv mysql here https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iii/buildmimic/mysql</li>
</ol>


<h2> Preprocessing the data </h2>
<ol>
    <li> create two folders in the root of the repository named raw_data and data, raw_data will contain data exported from the SQL tables and scripts we run in the following steps, while the data folder will be the recods from raw_data that have been through some additional python preprocessing</li>
    <li> We will first get the sapsii score for the baseline model. Run the following sql scripts in the following order in the /preprocess/saps folder. The following scripts are copied from the mimic iii repo, some of them are refactored and changed to suit our use case https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iii/</li>
    <ol>
        <li> ventilation_classification.sql </li>
        <li> ventilation_durations.sql </li>
        <li> ventilation_first_day.sql </li>
        <li> blood_gas_first_day.sql </li>
        <li> blood_gas_first_day_arterial.sql </li>
        <li> vitals_first_day.sql </li>
        <li> labs_first_day.sql </li>
        <li> gcs_first_day.sql </li>
        <li> urine_output_first_day.sql </li>
        <li> custom_sapsii.sql </li>
    </ol>
    <li> export the sapsii_logistic_regression_label table into a CSV file using DBViewer or via mysql commands to a file in the directory /raw_data/sapsii_logistic_regression_label.csv this data will become the labels for our baseline model</li>
    <li> export the sapsii_score_maxed table into a CSV file using DBViewer or via mysql commands to a file in the directory /raw_data/sapsii_feature_scores_maxed.csv this data will be the features for our baseline model </li>
    <li> We will next run the sql scripts to export data for our attention and RNN models. go to the ./preprocess/ folder </li>
    <li> run the extract_final_mimic.sql file (this can take ~8-12 hours depending on your CPU spec</li>
    <li> extract the KY_MIMIC_EVENTS_V4 table into a CSV file in the directory /raw_data/MIMIC_FULL_BATCH.csv</li>
    <li> extract the KY_ADM_LENGTH table into a CSV file in the directory /raw_data/MIMIC_ADM_INFO.csv </li>
    <li> go to the /preprocess folder again and run <code> python3 split_data.py </code> </li>
    <li> go to the /preprocess folder again and run <code> python3 aggregation.py </code> </li>
</ol>

<hr>


<h2> Running Baseline Model </h2>
<hr>
Make sure that all the preprocessing code was run succesfully, afterwards you just need to go to the <code> baseline </code> folder and run the following <code> python3 main.py </code>



<h2> Running Attention and RNN Model </h2>
<hr>






<h2> Results </h2>
<hr>

<table>
    <tr>
        <th> Model </th>
        <th> AUROC </th>    
        <th> AUPRC </th>
    </tr>
    <tr>
        <td> Baseline </td>
        <td> 0.808 </td>
        <td> 0.410 </td>
    </tr>
    <tr>
        <td> Baseline </td>
        <td> 0.7767 </td>
        <td> 0.1711 </td>
    </tr>
    <tr>
        <td> Baseline </td>
        <td> 0.8421 </td>
        <td> 0.2311 </td>
    </tr>
    <tr>
        <td> Baseline </td>
        <td> 0.8749 </td>
        <td> 0.2782 </td>
    </tr>
</table>

<h2>


<h2> Code References </h2>

<ol>
    <li> https://github.com/zhmd/monitor_icu </li>
    <li> https://github.com/MIT-LCP/mimic-code/ </li>
</ol>