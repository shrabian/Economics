"""
A simulation of a theoretical labour market as taught at The University of
Melbourne economics department in Intermediate Macroeconomics.

Implemented by Sharan K.
"""

from typing import List, Tuple, Union
from math import e
import matplotlib.pyplot as plt
import numpy as np

TOLERANCE = 1.0e-5

class InvalidValueError(Exception):
    def __init__(self, msg):
        self.msg = msg


class BrokenMarketError(Exception):
    def __init__(self, msg):
        self.msg = msg


class Shock:
    """
    A representation of a shock
    :param new_val: value to be assumed by a Shockable object after a shock
    :param time: time of shock execution
    """
    def __init__(self, new_val: Union[int, float], time: int):
        self.new_val = new_val
        self.time = time

class Shockable:
    """
    A respresentation of exogenous, shockable variables in the labour market.
    :param val: the current value assumed by the variable
    :param shocks: a list of all the shocks scheduled for the variable
    """
    def __init__(self, val: Union[int, float]):
        self.val = val
        self.shocks = []

    def schedule_shock(self, new_val: Union[int, float], time: int):
        """
        Creates a new shock object to execute at 'time' and overrides
        any other shock scheduled for the same time
        :param new_val: value to be assumed by this object after shock
        :param time: shock execution time
        """
        # Remove other shocks scheduled for variable at 'time'
        for i in range(len(self.shocks)):
            if self.shocks[i].time == time:
                del(self.shocks[i])

        # Schedule shock
        shock = Shock(new_val, time)
        self.shocks.append(shock)


    def check(self, time: int) -> Tuple[Union[int, float]]:
        """
        Execute any shocks scheduled for 'time' by setting 'val' attribute
        to the shock objects 'new_val'
        :param time: current market time
        :return: a tuple consisting of the variables previous and new value
        """
        for shock in self.shocks:
            if shock.time == time:
                prev_val = self.val
                self.val = shock.new_val
                return (prev_val, self.val)

    def validate(self, name: str,
                       lower: Union[int, float],
                       upper: Union[int, float] = np.inf,
                       inclusive: bool = False):

        """
        Verifies this object's 'val' attribute is within specified bounds and
        raises and InvalidValueError otherwise
        :param name: variable representation (e.g. 'j')
        :param lower: lower bound of value
        :param upper: upper bound of value
        :param inclusive: the inclusion rule for the bounds
        """
        output = f"'{name}' out of bounds: '{name}' = {self.val}"
        if inclusive:
            if not lower <= self.val <= upper:
                raise InvalidValueError(output)

        else:
            if not lower < self.val < upper:
                raise InvalidValueError(output)

    def equals(self, shockable):
        if abs(self.val - shockable.val) <= TOLERANCE:
            return True
        return False


class Matcher:
    """
    A representation of the unemployed to vacancy matching process in
    the labour market
    :param A: market matching efficiency as a Shockable
    :param a: relative share of vacancy to unemployment in match making
    as a Shockable
    :param tightness: the labour market tightness (u/v)
    """
    def __init__(self, A,
                       a,
                       j: Union[int, float],
                       c: Union[int, float]):
        self.A = A
        self.a = a
        self.tightness = self.optimal_tightness(j, c)

    def update(self, firm):
        """
        Updates tightness given firm's vacancy fill value and job posting cost.
        :param firm: the market's firm side
        self.tightness = self.optimal_tightness(firm.j.val, firm.c.val)
        """
        self.tightness = self.optimal_tightness(firm.j.val, firm.c.val)

    def optimal_tightness(self, j: Union[int, float],
                                c: Union[int, float]) -> float:
        return (self.A.val * j / c)**(1 / (1 - self.a.val))

    def get_frate(self) -> float:
        """Returns the job finding rate given market conditions"""
        rate = self.A.val * self.tightness**self.a.val
        return rate

    def check_shock(self, time: int) -> List[Tuple]:
        """
        Checks for shocks in exogenous variables (A, a) and validates their
        new values
        :param time: market time
        :return: list of tuples which represent a shock if one has occured
        """
        A_shock = self.A.check(time)
        a_shock = self.a.check(time)

        self.A.validate('A', 0, 1)
        self.a.validate('a', 0, 1)
        return [("A", A_shock), ("a", a_shock)]

    def equals(self, matcher):
        if self.A.equals(matcher.A) and\
           self.a.equals(matcher.a) and\
           abs(self.tightness - matcher.tightness) <= TOLERANCE:
            return True

        return False


class FirmSide:
    """
    A representation of the firm side of the labour market.
    :param j: value derived from filling a vacancy as a Shockable
    :param c: cost of posting a vacancy as a Shockable
    :param v: list of vacancy rate across time
    """
    def __init__(self, j, c, matcher, labour):
        self.j, self.c = j, c
        self.q = matcher.get_frate() / matcher.tightness
        self.v = [matcher.tightness * labour.u[-1]]

    def update(self, labour, matcher):
        """
        Updates vacancy via the job creation curve
        :param labour: LabourSide object of the labour market
        :param matcher: Matcher object of the labour market
        """
        self.q = matcher.get_frate() / matcher.tightness
        vacancies = matcher.tightness * labour.u[-1]
        self.v.append(vacancies)

    def check_shock(self, time):
        """
        Checks for shocks in exogenous variables (j, c) and validates their
        new values
        :param time: market time
        :return: list of tuples which represent a shock if one has occured
        """
        j_shock = self.j.check(time)
        c_shock = self.c.check(time)

        self.j.validate('j', 0)
        self.c.validate('c', 0)

        return [("j", j_shock), ("c", c_shock)]

    def vacancy_duration(self):
        return 1 / self.q

    def equals(self, firms):
        if self.j.equals(firms.j) and\
           self.c.equals(firms.c) and\
           abs(self.v[-1] - firms.v[-1]) <= TOLERANCE:
            return True

        return False


class LabourSide:
    """
    A representation of the labour side of the labour market
    :param s: job seperation rate as a Shockable
    :param f: job finding rate as a Shockable
    :param u: list of unemployment rates across time
    """
    def __init__(self, s, matcher):
        self.s = s
        self.f = matcher.get_frate()
        self.u = [self.steady_state()]

    def update(self, matcher):
        """
        Updates unemployment via the beveridge curve
        :param matcher: Matcher object of the labour market
        """
        # Inquires and updates f which allows unemployment to flow
        self.f = matcher.get_frate()
        unemployment = self.s.val + (1 - self.s.val - self.f) * self.u[-1]
        self.u.append(unemployment)

    def steady_state(self):
        """Returns steady state unemployment"""
        return self.s.val / (self.s.val + self.f)

    def fluidity_measure(self):
        """
        Measures market fluditiy by taking average of job finding and
        job seperating rate
        """
        return (self.s.val + self.f) / 2

    def check_shock(self, time: int):
        """
        Checks for shocks in exogenous variable 's', validates
        its new value and checks if unemployment flow is still stable
        :param time: market time
        :return: list of tuples which represent a shock if one has occured
        """
        s_shock = self.s.check(time)
        self.s.validate('s', 0, 1, True)
        self.check_stable_flow(time)

        return [("s", s_shock)]

    def check_stable_flow(self, time: int):
        """
        Checks if the market has broken, which occurs when s+f>=1, and
        raises an error if this condition is met.
        :param time: market time
        """
        msg = f"Shock at t = {time-1} was valid but broke the market. s+f >= 1"
        if self.s.val + self.f >= 1:
            raise BrokenMarketError(msg)

    def unemployment_duration(self):
        return 1/self.f

    def equals(self, labour):
        if self.s.equals(labour.s) and\
           abs(self.f - labour.f) <= TOLERANCE and\
           abs(self.u[-1] - labour.u[-1]) <= TOLERANCE:
            return True

        return False


class LabourMarket:
    """
    A representation of a theoretical labour market subject to flows and shocks
    :param matcher: match maker in the market
    :param labour: labour side of market
    :param firms: firms side of market
    :param shock_log: dictionary log of all shocks that have already occured
    :param time: time in the market since its creation
    """
    def __init__(self, s: Union[int, float] = 0.04,
                       j: Union[int, float] = 3,
                       c: Union[int, float] = 1,
                       A: Union[int, float] = 0.5,
                       a: Union[int, float] = 0.5,
                ):
        # Convert parameters into 'Shockable' objects
        s, j, c = Shockable(s), Shockable(j), Shockable(c)
        A, a = Shockable(A), Shockable(a)

        # Order of initialisation matters
        self.matcher = Matcher(A, a, j.val, c.val)
        self.labour = LabourSide(s, self.matcher)
        self.firms = FirmSide(j, c, self.matcher, self.labour)

        # Time initalised to -1 to allow shocks to occur at t = 0
        self.time = -1
        self.shock_log = {}

    def update(self):
        """
        Updates the labour market by one period, capturing all
        flow and shock affects
        """
        # A labour-side lag. Shocks occur after labour updates
        self.labour.update(self.matcher)

        # Checks for shocks and updates shock_log accordingly
        self.shock_update()
        self.time += 1

        # Matching institutions and firms are informed of shocks
        self.matcher.update(self.firms)
        self.firms.update(self.labour, self.matcher)

    def shock_update(self):
        """
        Checks shock for each market object and appends the returned tuple
        to shock_log
        """
        curr_shocks = [
                        self.labour.check_shock(self.time),
                        self.matcher.check_shock(self.time),
                        self.firms.check_shock(self.time)
                      ]

        # Each check_shock returns a list of tuples, hence the nested loop
        self.shock_log[self.time] = []
        for shock_list in curr_shocks:
            for shock in shock_list:
                if shock[1]:
                    new_rep = (shock[0], shock[1][0], shock[1][1])
                    self.shock_log[self.time].append(new_rep)

    def fast_forward(self, T: int):
        """
        Runs the market till some time in the future
        :param T: time to fast forward to
        """
        for i in range(T):
            self.update()

    def shock_j(self, new_val: Union[int, float], time: int):
        self.firms.j.schedule_shock(new_val, time)

    def shock_c(self, new_val: Union[int, float], time: int):
        self.firms.c.schedule_shock(new_val, time)

    def shock_s(self, new_val: Union[int, float], time: int):
        self.labour.s.schedule_shock(new_val, time)

    def shock_A(self, new_val: Union[int, float], time: int):
        self.matcher.A.schedule_shock(new_val, time)

    def shock_a(self, new_val: Union[int, float], time: int):
        self.matcher.a.schedule_shock(new_val, time)

    def get_udata(self):
        # Returns list of unemployment rates over time till present
        return self.labour.u.copy()

    def get_vdata(self):
        # Returns list of unemployment rates over time till present
        return self.firms.v.copy()

    def get_tightness(self):
        # Returns current market tightness
        return self.matcher.tightness

    def time_series(self, label1: str = "Unemployment",
                          label2: str = "Vacancies",
                          x_label: str = "Time",
                          y_label: str = "Rate"):
        """
        Plots unemployment and vacancy rates over time till present
        """
        fig, ax = plt.subplots()
        time = np.arange(-1, self.time + 1)
        udata = self.get_udata()
        vdata = self.get_vdata()
        ax.plot(time, udata, label = label1)
        ax.plot(time, vdata, label = label2)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        plt.legend()
        plt.show()

    def equals(self, market):
        if self.labour.equals(market.labour) and\
           self.firms.equals(market.firms) and\
           self.matcher.equals(market.matcher):
            return True

        return False

#Tasks

#Think about how to distribute functions: i.e. should the user of code
# have to call econ.labour to access the steady state or just econ

#Edit graphing functions so that users get to choose what data is displayed
#e.g. vacancy rate or unemployment rate, as well as what to label axes

#Add tests


if __name__ == "__main__":
    econ = LabourMarket()
    econ.shock_j(1000, 1)
    econ.shock_c(700,1)
    econ.shock_a(0.4, 20)
    econ.shock_A(0.7, 40)
    econ.shock_s(0.1, 60)
    econ.shock_j(1200, 60)

    print("\nSteady state unemployment rate log")
    print(f"Steady state at t={econ.time}: {econ.labour.steady_state()}")
    econ.fast_forward(20)
    print(f"Steady state at t={econ.time}: {econ.labour.steady_state()}")
    econ.fast_forward(20)
    print(f"Steady state at t={econ.time}: {econ.labour.steady_state()}")
    econ.fast_forward(20)
    print(f"Steady state at t={econ.time}: {econ.labour.steady_state()}")
    econ.fast_forward(20)
    print(f"Steady state at t={econ.time}: {econ.labour.steady_state()}")

    print("\nShock Log")
    for year in econ.shock_log:
        for shock in econ.shock_log[year]:
            print(f"Shock at t={year}: {shock[0]} jumps from {shock[1]} to {shock[2]}")

    econ.time_series()
