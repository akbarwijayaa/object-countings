import io
import os
import cv2
import psycopg2
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine
from configparser import ConfigParser
from os.path import join, dirname, abspath

from matplotlib import pyplot as plt
from matplotlib import patches

base_path = dirname(abspath(__file__))

config = ConfigParser(interpolation=None)
config.read(join(base_path, 'config.ini'))

def update_api_status(api_id, status):
    table = config['table']['service']
    query = f"UPDATE {table} SET status = '{status}' WHERE id_api = {api_id};"
    
    try:
        connection = psycopg2.connect(
            user=config['auth']['user'],
            password=config['auth']['password'],
            host=config['auth']['host'],
            database=config['auth']['database']
        )
        
        connection.autocommit = True 
        
        with connection.cursor() as cursor:
            cursor.execute(query)
        
        connection.commit()
        
    except (Exception, psycopg2.Error) as error:
        print("Error while connecting to PostgreSQL:", error)
    
    finally:
        if connection:
            connection.close()

def connect_db():
     # Create Engine Database
    username = config['auth']['user']
    password = config['auth']['password']
    host = config['auth']['host']
    database_name = config['auth']['database']
    
    machine = create_engine(config['connection']['engine'] % (username, password, host, database_name))
    conn = machine.raw_connection()
    conn.set_client_encoding(config['connection']['encoding'])
    conn.set_client_encoding(config['connection']['encodings'])
    cursor = conn.cursor()
    cursor.execute("SET SCHEMA 'mb'")
    
    return conn, cursor

def append_data(table_name, model, **kwargs):
    columns = []
    data = [[]]
    for key , value in kwargs.items() :
        columns.append(key),
        data[0].append(value)
        
    df =  pd.DataFrame(data, columns = columns)
    df.request_date = pd.to_datetime(df.request_date , format = 'Y%-m%-d%  H%:M%:S%')
    
    if (table_name == config['table']['analytics'] and model == 'cage_wheel_track_double'):
            df.pred_transact = df.pred_transact * 2
    
    conn, cursor = connect_db()
    append_data_core(df, conn, cursor, table_name, df.columns)
    max_id = get_max_id(table_name, cursor)
    
    cursor.close()
    conn.close()
    return max_id

def append_data_core(df, conn, cursor, table_name, columns):
    conn.autocommit = True 
    output = io.StringIO()
    df.to_csv(output, sep='\t', header=False, index=False)
    output.seek(0)
    contents = output.getvalue()
    cursor.copy_from(output, '{}'.format(table_name), columns=df.columns, null="")
    conn.commit()


def get_max_id(table_name, cursor):
    if table_name == config['table']['analytics']:
        query = '''SELECT currval(pg_get_serial_sequence('analytics_counting_object', 'id_counting'));'''
        cursor.execute(query)
        max_id = cursor.fetchone()[0]
    else:
        max_id = None
    return max_id