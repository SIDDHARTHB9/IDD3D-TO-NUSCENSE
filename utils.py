import os
import json
import logging
import threading

log = logging.getLogger(__name__)
json_file_lock = threading.Lock()


def append_to_json_list(file_path, new_data_list):
    if not new_data_list:
        log.info(f"No new data to append to {os.path.basename(file_path)}")
        return

    with json_file_lock:
        existing_data = []
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        log.warning(f"{file_path} is not a list. Overwriting.")
                        existing_data = []
            except json.JSONDecodeError:
                log.warning(f"{file_path} is corrupted. Overwriting.")
                existing_data = []
        
        final_data = existing_data + new_data_list
        
        try:
            with open(file_path, 'w') as f:
                json.dump(final_data, f, indent=2)
            log.info(f"Appended {len(new_data_list)} items to {os.path.basename(file_path)}. Total: {len(final_data)}")
        except Exception as e:
            log.error(f"FATAL: Could not write to {file_path}: {e}")
            raise


def merge_and_overwrite_json_list(file_path, new_entries, key_field):
    with json_file_lock:
        existing_data = {}
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    existing_list = json.load(f)
                    if isinstance(existing_list, list):
                        for entry in existing_list:
                            if key_field in entry:
                                existing_data[entry[key_field]] = entry
                    else:
                        log.warning(f"{file_path} is not a list. Overwriting.")
            except json.JSONDecodeError:
                log.warning(f"{file_path} is corrupted. Overwriting.")
        
        for entry in new_entries:
            if key_field in entry:
                existing_data[entry[key_field]] = entry
        
        final_list = list(existing_data.values())
        
        try:
            with open(file_path, 'w') as f:
                json.dump(final_list, f, indent=2)
            log.info(f"Merged/overwrote {os.path.basename(file_path)}. Total items: {len(final_list)}")
        except Exception as e:
            log.error(f"FATAL: Could not write to {file_path}: {e}")
            raise


def load_json_safely(file_path, default=None):
    if default is None:
        default = []
        
    if not os.path.exists(file_path):
        return default
    
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        log.warning(f"Could not parse {file_path}. Returning default.")
        return default
    except Exception as e:
        log.error(f"Error reading {file_path}: {e}")
        return default


def save_json_safely(file_path, data, indent=2):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=indent)
        log.debug(f"Saved {os.path.basename(file_path)}")
    except Exception as e:
        log.error(f"FATAL: Could not write to {file_path}: {e}")
        raise


def ensure_directory(directory_path):
    try:
        os.makedirs(directory_path, exist_ok=True)
        log.debug(f"Ensured directory exists: {directory_path}")
    except Exception as e:
        log.error(f"Could not create directory {directory_path}: {e}")
        raise