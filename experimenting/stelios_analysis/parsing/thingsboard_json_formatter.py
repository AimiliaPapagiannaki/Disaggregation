import json
import time

# measurement identifiers
GENERIC_CLOUD_IDENTIFIERS_OLD = (
    ('efrq',),
    ('pwrA', 'pwrB', 'pwrC'),
    ('vltA', 'vltB', 'vltC'),
    ('curA', 'curB', 'curC'),
    ('rpwrA', 'rpwrB', 'rpwrC'),
    ('cosA', 'cosB', 'cosC'),
    ('svltA', 'svltB', 'svltC'),
    ('scurA', 'scurB', 'scurC'),
    ('scosA', 'scosB', 'scosC'),
    ('cnrgA', 'cnrgB', 'cnrgC'),
    ('pnrgA', 'pnrgB', 'pnrgC'),
    ('rsumA', 'rsumB', 'rsumC'),
    ('rssi',),
    ('cminA', 'cminB', 'cminC', 'cmaxA', 'cmaxB', 'cmaxC'),
    ('unused',),
    ('vminA', 'vminB', 'vminC', 'vmaxA', 'vmaxB', 'vmaxC')
)
GENERIC_CLOUD_IDENTIFIERS_OLD_1PHASE = (
    ('efrq',),
    ('pwrA',),
    ('vltA',),
    ('curA',),
    ('rpwrA',),
    ('cosA',),
    ('svltA',),
    ('scurA',),
    ('scosA',),
    ('cnrgA',),
    ('pnrgA',),
    ('rsumA',),
    ('rssi',),
    ('cminA', 'cmaxA',),
    ('unused',),
    ('vminA', 'vmaxA',)
)
KEYS_TO_CHANGE_CUR_OLD = ('scurA', 'scurB', 'scurC')
KEYS_TO_CHANGE_VLT_OLD = ('svltA', 'svltB', 'svltC')
KEYS_TO_CHANGE_COS_OLD = ('scosA', 'scosB', 'scosC')
ALT_KEYS_CUR_OLD = ('curA', 'curB', 'curC')
ALT_KEYS_VLT_OLD = ('vltA', 'vltB', 'vltC')
ALT_KEYS_COS_OLD = ('cosA', 'cosB', 'cosC')
COMPACT_VOLTAGE_MULTIPLIER_INVERSE = 100
COMPACT_CURRENT_MULTIPLIER_INVERSE = 10
COMPACT_POWER_F_MULTIPLIER_INVERSE = 100


def thingsboard_json(device_name, timestamp, measurement_dict):
    return json.dumps({device_name: [{'ts': timestamp, 'values': measurement_dict}]})

    """Creates a ThingsBoard JSON string from the measurement dict.

	The measurement dict is of the form {'vltA': 240.531, 'curA': 1.555}.
	Ask for the allowed identifiers. In particular, "compact" identifiers like scosA are NOT allowed.
	You will need to make the relevant transforamtion and use a regular identifier like cosA."""
