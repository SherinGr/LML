# This file contains a library to implement my trading strategy
import indicators as ind


class Strategy:
    # Add indicators to pass into the functions
    def __init__(self):
        """ PRE DEFINED CONSTANTS """
        self.time_frame = '1h'
        self.stoch_threshold = 10
        self.stoch_threshold_15m = 30

        """ SET UP THE INDICATORS """
        EMA13 = ind.ExponentialMovingAverage(window_length=13, time_frame=time_frame)
        ATRChannels26 = ind.ATRChannels(window_length=26, time_frame=time_frame)
        BB80 = ind.BollingerBand(window_length=80, time_frame=time_frame, tp_style='hlc3')
        SRSI = ind.StochasticRSI(stoch_length=14, time_frame=time_frame)
        SRSI15m = ind.StochasticRSI(stoch_length=14, time_frame='15m')
        SRSI5m = ind.StochasticRSI(stoch_length=14, time_frame='5m')

    def check_stochastic(self, time_frame):
        trend = get_trend()
        stoch = self.srsi.history.tail(1)
        thr = self.stoch_threshold

        if trend > 0 and srsi < thr:
            return True
        elif trend < 0 and srsi > 100 - thr:
            return True
        elif trend == 0 and thr < srsi < 100 - thr:
            return True
        else:
            return False


    def check_bb(BB):
        pass

    def check_atrchannel(ATR, EMA):
        pass

    def check_aov(EMAshort, EMAlong):
        pass

    def is_bullrun(self):
        pass

    def is_bearrun(self):
        pass

    def get_trend(self):
        pass
        # IDEA: Sum of the slopes (+1,-1) of 26 EMA and 80 MA
        # use MA to determine the slopes of the MA's!
        # (new indicator that needs to be programmed!)
        # return +1,0 or -1

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

    def back_test(self):
        # load data of all time frames
        # batch fit all the indicators
        # loop strategy over the data output
        pass


    def run_online():
        # run the strategy in real time and send signals to user when opportunity appears (or automatically trade)
        pass
