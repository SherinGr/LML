# This file contains a library to implement my trading strategy


def check_srsi(SRSI, threshold=10):
    pass


def check_bb(BB):
    pass


def check_atrchannel(ATR,EMA):
    pass


def check_aov(EMAshort, EMAlong):
    pass


def is_bullrun():
    pass


def is_bearrun():
    pass


def check_for_entry():
    pass
    # Check SRSI first on 1h chart

    # Check if price is in run:

    # if yes, check for pullback to AOV

    # if no, check for price outside +2 ATR (& BB?)

    # SOME MORE STUFF

    # If entry conditions met, check SRSI on 15min (loose threshold!)
    # If good, check SRSI on 5 min (strict threshold)


def check_for_exit():
    pass


def back_test(data):
    pass
    # test the strategy on some historical data


def run_online():
    # run the strategy in real time and send signals to user when opportunity appears (or automatically trade)
    pass
