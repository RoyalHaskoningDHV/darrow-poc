import os
import re
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
import datetime
import logging

from .nrr import collect_regenradar_data


logger = logging.getLogger(__name__)


class RawData():    
    
    def __init__(self):

        self.data_raw_dir = Path("./data/raw_immutable")
        self.data_raw_sub_dir = Path("friedhelm_hackert_lanuv_nrw_de")

        self.data_preproc_dir = Path('./data/preprocessed')
        
        self.shapefile_dir = Path('./data/preprocessed/areas/stroomgebieden')
        self.shapefile_name = Path('Roer_merged.shp')
        
        self.fews_filename = Path('DataFEWS.xlsx')
        
    def _get_discharge_file_paths(self):
        """Return all relevant discharge filepaths. We assume the data is
        in the roer/data/ folder.

        Returns
        -------
        files : list
            List of full filenames for all relevant discharge data.
        """
        files = list(
            (self.data_raw_dir / self.data_raw_sub_dir).glob("*_q_*minutenmittel*.csv")
        )
        files.append(Path(
            'data/raw_immutable/20200625_TSB und Abflüsse in der Rur_WL-Bartussek.xlsx'
        ))
        files.append(self.data_raw_dir / self.fews_filename)
        files.append(Path('data/raw_immutable/Jülich_WL_debiet.xlsx'))
        
        return files

    def _read_raw_csv_header(self, filename):
        """Extract header parameters as dictionary keys and following
        phrases as values (strings).

        Parameters
        ----------
        filename : path-like
            File to be loaded as binary file and then parsed with encoding
        encoding : str (default='cp1252')
            How to decode the binary information in file

        Returns
        -------
        df : pd.DataFrame
            Columns are header labels. There is a single row of values.
        """
        df = pd.read_csv(filename, nrows=11, header=None, sep=';', encoding='cp1252')
        df = df.T
        df.columns = df.iloc[0, :]
        df = df.drop(0)

        return df

    def _read_raw_obermaubach_data(self, filename):
        """Read raw discharge data from obermaubach .xlsx file.
        
        Parameters
        ----------
        filename : path-like
            Full or relative filepath of .xlsx file
        
        Returns
        -------
        pd.DataFrame
        """
        xl_file = pd.ExcelFile(filename)
        dfs = {
            sheet_name: xl_file.parse(sheet_name) for sheet_name in xl_file.sheet_names
        }
        df = dfs['Tabelle1'].iloc[9:]
        df.loc[:, df.columns[0]] = [str(x) for x in df.loc[:, df.columns[0]]]

        return df

    def _read_supplementary_juelich_data(
        self, 
        filename='data/raw_immutable/Jülich_WL_debiet.xlsx'
    ):
        """Read raw discharge data from .xlsx file.
        
        Parameters
        ----------
        filename : path-like
            Full or relative filepath of .xlsx file
        
        Returns
        -------
        pd.DataFrame
        """
        xl_file = pd.ExcelFile(filename)
        dfs = {
            sheet_name: xl_file.parse(sheet_name) for sheet_name in xl_file.sheet_names
        }
        df = dfs['Blad1'].iloc[5:]
        df.loc[:, df.columns[0]] = [str(x) for x in df.loc[:, df.columns[0]]]

        return df
    
    def _read_raw_csv_data(self, filename):
        """Read raw discharge data from .csv file.
        
        Parameters
        ----------
        filename : path-like
            Full or relative filepath of .csv file
        
        Returns
        -------
        pd.DataFrame
        """
        return pd.read_csv(  # The first 11 columns are irrelevant header information
            filename, 
            skiprows=11, 
            header=None,
            sep=';', 
            encoding='cp1252'
        )

    def _read_lanuv_from_fews(self):
        """Read recent LANUV data from FEWS .xlsx file.
        
        Returns
        -------
        df : pd.DataFrame
        """
        xl_file = pd.ExcelFile(self.data_raw_dir / self.fews_filename)
        dfs = {
            sheet_name: xl_file.parse(sheet_name) for sheet_name in xl_file.sheet_names
        }

        df = dfs['LANUV']
        ncols = df.shape[1]
        df = df.iloc[:, [0] + [i for i in range(1, ncols, 4)]]
        df.columns = df.loc[1, :]
        df = df.drop(index=[0, 1, 2])

        # Adjust column names to fit with what model expects
        df = df.rename(columns = {
            np.nan: 'TIME',
            'Altenburg_1': 'altenburg1',
            'Herzogenrath_1': 'herzogenrath1',
            'Herzogenrath_2': 'herzogenrath2',
            'Kirchberg1': 'kirchberg',
            'KornelimuensterW': 'kornelimuenster',
            'Linnich': 'linnich1',
            'Roer, Jülich': 'juelich',
            'Worm, Randerath': 'randerath',
            'Zerkall': 'zerkall1',
        })
        
        return df
        
    def _determine_datetime_format(self, dt_string):
        """Given an example string of a datetime string (e.g. '08.03.2020 08:00:00'),
        determine the correct format str to coerce str to datetime.
        
        Parameters
        ----------
        dt_string : str
        
        Returns
        -------
        format_string : str
            Datetime format string used e.g. by pd.to_datetime
        """
        if type(dt_string) != str:
            raise TypeError('Input has to be a string')
        
        if re.match('[0-9][0-9][0-9][0-9]-[0-9]+-[0-9]+ [0-9]+:[0-9]+:[0-9]+', dt_string):
            format_string = '%Y-%m-%d %H:%M:%S'
            
        elif re.match('[0-9]+-[0-9]+-[0-9][0-9][0-9][0-9] [0-9]+:[0-9]+:[0-9]+', dt_string):
            format_string = '%d-%m-%Y %H:%M:%S'
            
        elif re.match('[0-9]+\.[0-9]+\.[0-9][0-9][0-9][0-9] [0-9]+:[0-9]+:[0-9]+', dt_string):
            format_string = '%d.%m.%Y %H:%M:%S'
            
        else:
            raise NotImplementedError(
                'Datetime format is not recognized.'
                'Did you add a new data source with a different datetime format?'
            )

        return format_string

    def _set_column_names_and_types(self, df):
        """Set TIME and VALUE column names and appropriate data types.
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw dataframe read from xlsx or csv file of dicharge data.
        
        Returns
        -------
        pd.DataFrame
            With the columns 'TIME' (datetime) and 'VALUE' (int)
        """
        df.columns = ['TIME', 'VALUE']

        if type(df['TIME'].iloc[0]) != datetime.datetime:
            df['TIME'] = pd.to_datetime(
                df['TIME'],
                format=self._determine_datetime_format(df['TIME'].iloc[0])
            )
        
        if type(df['VALUE'].iloc[0]) == str:
            df['VALUE'] = [np.float64(str(v).replace(',', '.').strip()) for v in df['VALUE']]
        else:
            df = df.astype({'VALUE': np.float64})
        
        return df
    
    def _read_raw_stah_data(self, filename):
        """Read `Stah` location data from raw FEWS data dump.

        Parameters
        ----------
        file : pathlib.PosixPath
        
        Returns
        -------
        pd.DataFrame
            With two columns: datetime and discharge values.
        """
        xl_file = pd.ExcelFile(filename)
        dfs = {
            sheet_name: xl_file.parse(sheet_name) for sheet_name in xl_file.sheet_names
        }

        df = dfs['WL'].iloc[4:,[0, 1]]
        df.loc[:, df.columns[0]] = [str(x) for x in df.loc[:, df.columns[0]]]

        return df

    def _read_raw_rimburg_data(self, filename):
        """Read `Rimburg` location data from raw FEWS data dump.
        Rimburg is in-between Herzogenrath and Randerath (worm river).

        Parameters
        ----------
        file : pathlib.PosixPath
        
        Returns
        -------
        pd.DataFrame
            With two columns: datetime and discharge values.
        """
        xl_file = pd.ExcelFile(filename)
        dfs = {
            sheet_name: xl_file.parse(sheet_name) for sheet_name in xl_file.sheet_names
        }

        df = dfs['WL'].iloc[4:,[0, 5]]
        df.loc[:, df.columns[0]] = [str(x) for x in df.loc[:, df.columns[0]]]

        return df

    def _read_discharge_data(self, file, fews_location='stah'):
        """Read and slighty preprocess discharge data from .csv or .xlsx
        file.
        
        Parameters
        ----------
        file : pathlib.PosixPath
        fews_location : str (default='stah')
            Read either `stah` or `rimburg` data.
        
        Returns
        -------
        pd.DataFrame
            With the columns 'TIME' (datetime) and 'VALUE' (int)
        """
        if file.suffix == '.csv':
            df = self._read_raw_csv_data(file)

        elif '20200625_TSB' in file.name:
            df = self._read_raw_obermaubach_data(filename=file)

        elif 'DataFEWS' in file.name:
            if fews_location == 'stah':
                df = self._read_raw_stah_data(filename=file)
            elif fews_location == 'rimburg':
                df = self._read_raw_rimburg_data(filename=file)
            else:
                raise ValueError('fews_location has to be `stah` or `rimburg`')

        elif 'Jülich_WL_debiet' in file.name:
            df = self._read_supplementary_juelich_data(filename=file)

        else:
            raise NotImplementedError('Only .csv and .xlsx files can be handled at the moment')
        
        df = self._set_column_names_and_types(df)

        return df

    def _extract_location(self, file):
        """Given a filename of a discharge dataset, extract the 
        name of the location of the measurement station.
        
        Parameters
        ----------
        file : pathlib.PosixPath
        
        Returns
        -------
        str
        """
        location = file.name.split('_')[0]
        
        if location == '20200625':
            location = 'obermaubach'
        if location == 'Jülich':
            location = 'juelich_wl'
        
        return location

    def _preprocess_discharge_data(self, discharge_files):
        """Write preprocessed discharge data to .csv files in appropriate
        folder.

        Parameters
        ----------
        discharge_files : list
            List of pathlib.PosixPath objects denoting filenames of
            discharge data.
        """
        save_dir = self.data_preproc_dir / Path('discharge')
        save_dir.mkdir(parents=True, exist_ok=True)

        for i, file in enumerate(discharge_files):

            filename = Path(self._extract_location(file) + '.csv')

            print('{0}/{1}: Preprocessing discharge data, writing to {2}'.format(
                i + 1, len(discharge_files), save_dir / filename)
            )
            
            if 'DataFEWS' in file.name:
                df_discharge_roer = self._read_discharge_data(file, fews_location='stah')
                df_discharge_roer.to_csv(save_dir / Path('stah.csv'))
                df_discharge_worm = self._read_discharge_data(file, fews_location='rimburg')
                df_discharge_worm.to_csv(save_dir / Path('rimburg.csv'))

            else:
                df_discharge = self._read_discharge_data(file)
                df_discharge.to_csv(save_dir / filename)
    
    def preprocess_discharge_data(self, testing=False):
        """Read discharge data and write to single .csv file with some
        marginal preprocessing.
        
        Parameters
        ----------
        testing : Bool (default=False)
            When True, only parse the first file to speed up testing.
        """
        discharge_files = self._get_discharge_file_paths()
        
        if testing is True:
            discharge_files = discharge_files[0:1]

        self._preprocess_discharge_data(discharge_files)
    
    def get_shapefile_roer(self):
        """Read shapefile of Roer region.
        """
        shapefile = gpd.read_file(self.shapefile_dir / self.shapefile_name)
        
        return shapefile

    def get_smoothed_shapefile_roer(self):
        """Read shapefile of Roer region and smooth edges to avoid very
        long polygons, which will lead to errors when used in the regenradar API.
        """
        shapefile = gpd.read_file(self.shapefile_dir / self.shapefile_name)
        shapefile['geometry'] = shapefile['geometry'].simplify(tolerance=500)
        
        return shapefile

    def get_precipitation_from_regenradar(
        self, 
        testing=False,
        start_dates = [
            '2010-01-01', '2010-06-01',
            '2011-01-01', '2011-06-01',
            '2012-01-01', '2012-06-01', 
            '2013-01-01', '2013-06-01',
            '2014-01-01', '2014-06-01',
            '2015-01-01', '2015-06-01',
            '2016-01-01', '2016-06-01', 
            '2017-01-01', '2017-06-01',
            '2018-01-01', '2018-06-01', 
            '2019-01-01', '2019-06-01',
            '2020-01-01', '2020-06-01',
            '2021-01-01', '2021-06-01',
        ],
        end_dates = [
            '2010-06-30', '2011-01-30', 
            '2011-06-30', '2012-01-30',
            '2012-06-30', '2013-01-30', 
            '2013-06-30', '2014-01-30', 
            '2014-06-30', '2015-01-30', 
            '2015-06-30', '2016-01-30',
            '2016-06-30', '2017-01-30', 
            '2017-06-30', '2018-01-30',
            '2018-06-30', '2019-01-30', 
            '2019-06-30', '2020-01-30',
            '2020-06-30', '2021-01-30',
            '2021-06-30', '2022-01-30',
        ]
    ):
        """Convenience function to read regenradar precipitation data from
        regenradar API- from 2010-01-01 to 2020-06-01 in half year steps.
        Saves the data in `./data/raw_immutable/regenradar.csv`
        
        Parameters
        ----------
        testing : bool (default=False)
            When true, use a much smaller time range. Useful for quick testing.
        start_dates : list
        end_dates : list
        """
        save_path = self.data_raw_dir / Path('regenradar.csv')

        if os.path.exists(save_path):
            os.remove(save_path)
        
        shapefile = self.get_smoothed_shapefile_roer()
        
        if testing is True:
            start_dates = ['2010-01-01', '2010-02-01']
            end_dates = ['2010-02-01', '2010-03-01']
            save_path = self.data_raw_dir / Path('regenradar_test.csv')

        for i, (start_date, end_date) in enumerate(zip(start_dates, end_dates)):

            logger.info('REGENRADAR API: Loading data for year {} to {}'.format(
                start_date, end_date)
            )

            df = collect_regenradar_data(
                shapefile,
                start_time='{}'.format(start_date),
                end_time='{}'.format(end_date),
                freq='1H',
                srs='epsg:28992'
            )

            df.to_csv(save_path, mode='a')

    def read_raw_precipitation_from_file(self, filename=None, testing=False):
        """Convenience function for reading raw precipitation data gathered
        from regenradar.
        
        Parameters
        ----------
        filename : path-like (default=None)
            When no filename is provided try to read default in
            self.data_raw_dir / Path('regenradar.csv')
        testing : bool (default=False)
            When testing set a specific default filename
        
        Returns
        -------
        precip : pd.DataFrame
        """
        if filename is None:
            filename = self.data_raw_dir / Path('regenradar.csv')
        
        if testing:
            filename = self.data_raw_dir / Path('regenradar_test.csv')

        precip = pd.read_csv(filename, index_col=0).drop_duplicates(
            subset=['TIME', 'ID']
        ).reset_index(drop=True)
        
        return precip

    def create_precip_csv_files_per_region(self):
        """Read the long format regenradar data from .csv file and save
        data separately for each region, given the .csv files the names
        of the regions.
        
        Saves in 'data/preprocessed/precip/nrr/{regions_name}.csv'
        """
        save_path = self.data_preproc_dir / Path('precip/nrr/')
        save_path.mkdir(parents=True, exist_ok=True)
        
        shapefile = self.get_shapefile_roer()

        rr = self.read_raw_precipitation_from_file()
        rr = rr.drop(index=rr.index[rr['ID'] == 'ID']).reset_index()

        rr_ids = rr['ID'].unique()
        for rr_id in rr_ids:
            
            rr_region = rr.loc[rr['ID'] == rr_id, :]

            save_name = "{}.csv".format(
                shapefile.loc[shapefile['ID'] == np.int64(rr_id), 'naam'].values[0]
            )
            rr_region.to_csv(save_path / Path(save_name))    

    def get_precipitation_surplus_from_knmi(
        self, 
        start_date='2001-11-01', 
        end_date='2022-01-01'
    ):
        """Convenience function to read precipitation and evaporation data
        from KNMI API, store in .csv file.
        
        Parameters
        ----------
        start_date : str (default='2001-11-01')
            Beginning of timerange from which to query data.
        end_date : str (default='2020-06-01')
            End of timerange from which to query data.
        """
        from sam.data_sources import read_knmi_stations, read_knmi_station_data

        save_path = self.data_raw_dir / Path('knmi')
        save_path.mkdir(parents=True, exist_ok=True)

        knmi_stations = read_knmi_stations()
        knmi_maastricht = knmi_stations.loc[knmi_stations['name'] == 'Maastricht', :]

        data_knmi = read_knmi_station_data(
            start_date=start_date,
            end_date=end_date,
            stations=knmi_maastricht['number'].values,
            variables=['EV24']  # neerslag, verdamping
        )
        
        data_knmi.to_csv(save_path / Path('evap.csv'))
    
    def read_precipitation_surplus_from_file(self, filename=None):
        """Convenience function for reading ]precipitation surplus data gathered
        from knmi and saved to local .csv file.
        
        Parameters
        ----------
        filename : path-like (default=None)
            When no filename is provided try to read default in
            self.data_raw_dir / Path('knmi') / Path('evap.csv')

        Returns
        -------
        precip_surp : pd.DataFrame
        """
        if filename is None:
            filename = self.data_raw_dir / Path('knmi') / Path('evap.csv')

        precip_surp = pd.read_csv(filename, index_col=0).reset_index(drop=True)

        return precip_surp
