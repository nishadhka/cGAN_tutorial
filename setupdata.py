import gc

from data import all_fcst_fields
from data.tfrecords_generator import DataGenerator
import numpy as np


# Incredibly slim wrapper around tfrecords_generator.DataGenerator.  Can probably remove...
def setup_batch_gen(train_years, batch_size=16, autocoarsen=False, weights=None):
    # print(f"autocoarsen flag is {autocoarsen}")
    batch_gen_train = DataGenerator(
        train_years, batch_size=batch_size, autocoarsen=autocoarsen, weights=weights
    )
    return batch_gen_train


def setup_full_image_dataset(years, batch_size=1, autocoarsen=False):
    from data import DataGenerator as DataGeneratorFull
    from data import get_dates

    dates = get_dates(years, start_hour=30, end_hour=54)
    #dates = [str(date).replace("-","")\
    #         for date in\
    #         np.arange("2023-05-01","2023-08-01",np.timedelta64(1,"D"),dtype="datetime64[D]")\
    #        ]
    data_full = DataGeneratorFull(
        dates=dates,
        fcst_fields=all_fcst_fields,
        start_hour=30,
        end_hour=54,
        batch_size=batch_size,
        log_precip=True,
        shuffle=True,
        constants=True,
        fcst_norm=True,
        autocoarsen=autocoarsen,
        consolidated=False,
    )
    return data_full


def setup_data(
    train_years=None, val_years=None, autocoarsen=False, weights=None, batch_size=None
):
    batch_gen_train = (
        None
        if train_years is None
        else setup_batch_gen(
            train_years=train_years,
            batch_size=batch_size,
            autocoarsen=autocoarsen,
            weights=weights,
        )
    )

    data_gen_valid = (
        None
        if val_years is None
        else setup_full_image_dataset(val_years, autocoarsen=autocoarsen)
    )

    gc.collect()
    return batch_gen_train, data_gen_valid
