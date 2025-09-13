import sys
from networksecurity.logging.logger import logging

def error_message_detail(error_msg, error_detail: sys):
    _1, _2, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = f"Error occured in python script name [{file_name}] line number [{line_number}] error message [{str(error_msg)}]"

    return error_message

class NetworkSecurityException(Exception):
    def __init__(self, error_msg, error_detail:sys):
        super().__init__(error_msg)
        self.error_msg = error_message_detail(error_msg=error_msg, error_detail=error_detail)

    def __str__(self):
        return self.error_msg
    
if __name__ == "__main__":
    try:
        logging.error("Error will occur")
        a=1/0
    except Exception as e:
        raise NetworkSecurityException(e, sys)