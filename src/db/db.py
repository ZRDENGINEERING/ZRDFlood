# db.py
import os
from sqlalchemy import create_engine, text, event

from sqlalchemy.pool import QueuePool
import geopandas as gpd
import fiona
import pandas as pd



def _env(key, default=''):
    return os.environ.get(key, default).strip()


engine = create_engine(
    f"postgresql://postgres:{_env('PGPASSWORD')}@localhost/zrdproj",
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=0,
    pool_timeout=10,
    connect_args={"connect_timeout": 10}
)

@event.listens_for(engine, "connect")
def set_search_path(dbapi_conn, connection_record):
    cursor = dbapi_conn.cursor()
    cursor.execute("SET search_path TO public")
    cursor.close()


def table_exists(table_name, schema):
    with engine.connect() as con:
        result = con.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = :schema
                AND table_name = :table
            );
        """), {"schema": schema, "table": table_name})
        row = result.fetchone()
        return bool(row[0]) if row is not None else False


def has_geometry(table_name, schema):
    with engine.connect() as con:
        result = con.execute(text("""
            SELECT EXISTS (
                SELECT FROM geometry_columns
                WHERE f_table_schema = :schema
                AND f_table_name = :table
            );
        """), {"schema": schema, "table": table_name})
        return result.fetchone()[0]


def has_features(table_name, schema):
    if not table_exists(table_name, schema):
        return False

    geom_col = get_geom_col(table_name, schema)

    with engine.connect() as con:
        if geom_col:
            result = con.execute(text(f"""
                SELECT EXISTS (
                    SELECT 1 FROM "{schema}"."{table_name}"
                    WHERE "{geom_col}" IS NOT NULL
                      AND NOT ST_IsEmpty("{geom_col}")
                    LIMIT 1
                );
            """))
        else:
            result = con.execute(text(f"""
                SELECT EXISTS (
                    SELECT 1 FROM "{schema}"."{table_name}" LIMIT 1
                );
            """))
        return result.fetchone()[0]



def get_geom_col(table_name, schema):
    with engine.connect() as con:
        result = con.execute(text("""
            SELECT f_geometry_column 
            FROM geometry_columns
            WHERE f_table_schema = :schema
            AND f_table_name = :table
            LIMIT 1;
        """), {"schema": schema, "table": table_name})
        row = result.fetchone()
        return row[0] if row else None


def get_table(table_name, schema):
    geom_col = get_geom_col(table_name, schema)

    if not geom_col:
        with engine.connect() as con:
            return pd.read_sql(f'SELECT * FROM "{schema}"."{table_name}";', con)

    with engine.connect() as con:
        gdf = gpd.read_postgis(
            f'SELECT * FROM "{schema}"."{table_name}";',
            con,
            geom_col=geom_col
        )

    if gdf.empty or gdf[geom_col].dtype != 'geometry':
        with engine.connect() as con:
            srid = con.execute(text(
                """
                SELECT srid FROM geometry_columns
                WHERE f_table_schema = :schema
                  AND f_table_name = :table_name
                  AND f_geometry_column = :geom_col
                """
            ), {'schema': schema, 'table_name': table_name, 'geom_col': geom_col}).scalar()
        gdf[geom_col] = gpd.GeoSeries(gdf[geom_col], crs=f'EPSG:{srid}' if srid else None)
        gdf = gdf.set_geometry(geom_col)

    return gdf



def create_table(table_name, schema, columns):
    """
    columns: str of column definitions, e.g.
        'id SERIAL PRIMARY KEY, geom GEOMETRY(Point, 2277), name TEXT'
    """
    with engine.connect() as con:
        con.execute(text(
            f'CREATE TABLE IF NOT EXISTS "{schema}"."{table_name}" ({columns});'
        ))
        con.commit()



def make_schema(proj_num):
    if proj_num is None or str(proj_num).strip() == '':
        raise ValueError("make_schema called with no proj_num — cfg failed to resolve a project")
    return f'_{proj_num}'


def ensure_schema(schema):
    with engine.connect() as con:
        con.execute(text('SET search_path TO public;'))
        con.execute(text('CREATE EXTENSION IF NOT EXISTS postgis;'))
        con.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}";'))
        con.commit()


def insert_table(gdf, table_name, schema, if_exists='replace', srid=None):
    ensure_schema(schema)
    try:
        gdf = gdf.drop(columns=['id'], errors='ignore')
    except Exception:
        pass

    if srid is None:
        if hasattr(gdf, 'crs') and gdf.crs is not None:
            srid = gdf.crs.to_epsg() or 4269
        else:
            srid = 4269

    geom_col = gdf.geometry.name if hasattr(gdf, 'geometry') else None

    if hasattr(gdf, 'geometry'):
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
        if gdf.empty:
            print(f'  insert_table: no valid geometries for {schema}.{table_name}, skipping.')
            return False

    with engine.connect() as con:
        con.execute(text('SET search_path TO public'))
        con.commit()

        if if_exists == 'replace' and table_exists(table_name, schema):
            con.execute(text(f'DROP TABLE "{schema}"."{table_name}"'))
            con.commit()
            gdf.to_postgis(table_name, con, schema=schema, if_exists='fail', index=False)
        else:
            gdf.to_postgis(table_name, con, schema=schema, if_exists=if_exists, index=False)

        con.execute(text(f'ALTER TABLE "{schema}"."{table_name}" ADD COLUMN IF NOT EXISTS id SERIAL PRIMARY KEY'))
        con.commit()

        if geom_col:
            try:
                con.execute(text(f"""
                    SELECT UpdateGeometrySRID(
                        '{schema}'::varchar, '{table_name}'::varchar, '{geom_col}'::varchar, {srid}
                    )
                """))
                con.commit()
            except Exception as e:
                print(f'  ⚠ UpdateGeometrySRID failed for {schema}.{table_name}.{geom_col}: {e}')
                con.rollback()

    return True




def insert_gpkg_layer(gpkg_path, layer_name, schema, table_name=None, if_exists='replace'):
    table_name = table_name or layer_name

    gdf = gpd.read_file(gpkg_path, layer=layer_name)
    insert_table(gdf, table_name or layer_name, schema, if_exists=if_exists)

    with engine.begin() as con:
        con.execute(text(
            f'ALTER TABLE "{schema}"."{table_name}" ADD COLUMN IF NOT EXISTS id SERIAL PRIMARY KEY'
        ))


def gpkg_layer_exists(gpkg_path, layer_name):
    return layer_name.lower() in [l.lower() for l in fiona.listlayers(gpkg_path)]


def get_table_mtime(table_name, schema):
    """Return MAX(modified_at) for a table, or None if missing."""
    query = text(f'SELECT MAX(modified_at) FROM "{schema}"."{table_name}"')
    try:
        with engine.connect() as con:
            row = con.execute(query).fetchone()
        return row[0] if row else None
    except Exception:
        return None


def get_srid(table_name, schema):
    with engine.connect() as con:
        result = con.execute(text("""
            SELECT srid FROM geometry_columns
            WHERE f_table_schema = :schema AND f_table_name = :table
            LIMIT 1
        """), {"schema": schema, "table": table_name})
        row = result.fetchone()
        return row[0] if row else None
    


if __name__ == "__main__":
    '''
        insert_table IS FOR BASIC PG LOADS OF GEODATAFRAME IN MEMORY, USE LOAD FILE IN gisdbutils.py FOR LARGE FILE BULK LOADS
    '''
    # insert_table(gdf, table_name, schema, if_exists='replace', srid=None)

