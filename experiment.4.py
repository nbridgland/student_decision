"""Run an experiment in which a portfolio of offers is deployed to a population."""

import logging
import unittest

import argparse
import os.path
import glob
import datetime
import numpy
import shutil

from externalities import World, Offer, Transaction, Event, Categorical
from utilities import ProfileGenerator, mkdir_if_missing
from person import Person
from population import Population

dt_fmt = '%Y%m%d'
now = datetime.datetime.now()


def create_people_0(n):
    profile_optout_rate = 0.1
    min_tenure = 0
    max_tenure = 365
    min_age = 24
    max_age = 36
    non_binary_fraction = 0.02
    min_income = 50000
    max_income = 75000
    beta = 1.0 / 0.0004
    g = lambda x: 1.0 / (1.0 + numpy.exp(-x))
    g_inv = lambda y: numpy.log(y / (1.0 - y))

    pg = ProfileGenerator()

    people = list()
    for i in range(n):
        became_member_on = (now - datetime.timedelta(days=numpy.random.choice(range(min_tenure, max_tenure)))).strftime(dt_fmt)

        if numpy.random.random() < 1.0 - profile_optout_rate:
            age, gender = pg.sample_age_gender(1, non_binary_fraction, min_age, max_age)
            dob = (now - datetime.timedelta(days=int(age*365.25 + numpy.random.choice(range(365))))).strftime(dt_fmt)
            income = None
            for i in range(25):
                x = max(25000, numpy.random.exponential(beta))
                if min_income <= x <= max_income:
                    income = x
                    break
        else:
            dob = None
            gender = None
            income = None

        person_view_offer_sensitivity = Categorical(['background', 'offer_age', 'web', 'email', 'mobile', 'social'],
                                                    [g_inv(0.20) - 4, -abs(g_inv(0.01) / float(1 * 24 * 30)), 0, 1, 1,
                                                     2])
        person_make_purchase_sensitivity = Categorical(
            ['background', 'time_since_last_transaction', 'last_viewed_offer_strength', 'viewed_active_offer'],
            [g_inv(1.0 / 24.0), abs(g_inv(0.10) / float(1 * 24 * 30)), 1, 1])
        person_purchase_amount_sensitivity = Categorical(
            ['background', 'income_adjusted_purchase_sensitivity', 'front page', 'local', 'entertainment', 'sports',
             'opinion', 'comics', 'sweet', 'sour', 'salty', 'bitter', 'umami'], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        people.append(Person(became_member_on,
                             dob=dob,
                             gender=gender,
                             income=income,
                             view_offer_sensitivity=person_view_offer_sensitivity,
                             make_purchase_sensitivity=person_make_purchase_sensitivity,
                             purchase_amount_sensitivity=person_purchase_amount_sensitivity))

    return people


def create_people_1(n):
    profile_optout_rate = 0.2
    min_tenure = 60
    max_tenure = 540
    min_age = 30
    max_age = 50
    non_binary_fraction = 0.04
    min_income = 30000
    max_income = 60000
    beta = 1.0 / 0.0004
    g = lambda x: 1.0 / (1.0 + numpy.exp(-x))
    g_inv = lambda y: numpy.log(y / (1.0 - y))

    pg = ProfileGenerator()

    people = list()
    for i in range(n):
        became_member_on = (now - datetime.timedelta(days=numpy.random.choice(range(min_tenure, max_tenure)))).strftime(dt_fmt)

        if numpy.random.random() < 1.0 - profile_optout_rate:
            age, gender = pg.sample_age_gender(1, non_binary_fraction, min_age, max_age)
            dob = (now - datetime.timedelta(days=int(age*365.25 + numpy.random.choice(range(365))))).strftime(dt_fmt)
            income = None
            for i in range(25):
                x = max(25000, numpy.random.exponential(beta))
                if min_income <= x <= max_income:
                    income = x
                    break
        else:
            dob = None
            gender = None
            income = None

        person_view_offer_sensitivity = Categorical(['background', 'offer_age', 'web', 'email', 'mobile', 'social'],
                                                    [g_inv(0.50) - 4, -abs(g_inv(0.01) / float(1 * 24 * 30)), 1, 1, 1,
                                                     1])
        person_make_purchase_sensitivity = Categorical(
            ['background', 'time_since_last_transaction', 'last_viewed_offer_strength', 'viewed_active_offer'],
            [g_inv(1.0 / 48.0), abs(g_inv(0.10) / float(1 * 24 * 30)), 1, 1])
        person_purchase_amount_sensitivity = Categorical(
            ['background', 'income_adjusted_purchase_sensitivity', 'front page', 'local', 'entertainment', 'sports',
             'opinion', 'comics', 'sweet', 'sour', 'salty', 'bitter', 'umami'], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        people.append(Person(became_member_on,
                             dob=dob,
                             gender=gender,
                             income=income,
                             view_offer_sensitivity=person_view_offer_sensitivity,
                             make_purchase_sensitivity=person_make_purchase_sensitivity,
                             purchase_amount_sensitivity=person_purchase_amount_sensitivity))

    return people


def create_people_2(n):
    profile_optout_rate = 0.2
    min_tenure = 60
    max_tenure = 540
    non_binary_fraction = 0.04
    min_age = 18
    max_age = None
    f_fraction = 0.67
    min_income = 50000
    max_income = 100000
    beta = 1.0 / 0.0004
    g = lambda x: 1.0 / (1.0 + numpy.exp(-x))
    g_inv = lambda y: numpy.log(y / (1.0 - y))

    pg = ProfileGenerator()

    people = list()
    for i in range(n):
        became_member_on = (now - datetime.timedelta(days=numpy.random.choice(range(min_tenure, max_tenure)))).strftime(dt_fmt)

        if numpy.random.random() < 1.0 - profile_optout_rate:
            if numpy.random.random() < f_fraction:
                age, gender = pg.sample_age_gender(1, non_binary_fraction, min_age, max_age, fixed_gender='F')
            else:
                age, gender = pg.sample_age_gender(1, non_binary_fraction, min_age, max_age, fixed_gender='M')
            dob = (now - datetime.timedelta(days=int(age*365.25 + numpy.random.choice(range(365))))).strftime(dt_fmt)
            income = None
            for i in range(25):
                x = max(25000, numpy.random.exponential(beta))
                if min_income <= x <= max_income:
                    income = x
                    break
        else:
            dob = None
            gender = None
            income = None

        person_view_offer_sensitivity = Categorical(['background', 'offer_age', 'web', 'email', 'mobile', 'social'],
                                                    [g_inv(0.20) - 4, -abs(g_inv(0.01) / float(1 * 24 * 30)), 1, 0, 1,
                                                     2])
        person_make_purchase_sensitivity = Categorical(
            ['background', 'time_since_last_transaction', 'last_viewed_offer_strength', 'viewed_active_offer'],
            [g_inv(1.0 / 48.0), abs(g_inv(0.10) / float(1 * 24 * 30)), 1, 1])
        person_purchase_amount_sensitivity = Categorical(
            ['background', 'income_adjusted_purchase_sensitivity', 'front page', 'local', 'entertainment', 'sports',
             'opinion', 'comics', 'sweet', 'sour', 'salty', 'bitter', 'umami'], [2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        people.append(Person(became_member_on,
                             dob=dob,
                             gender=gender,
                             income=income,
                             view_offer_sensitivity=person_view_offer_sensitivity,
                             make_purchase_sensitivity=person_make_purchase_sensitivity,
                             purchase_amount_sensitivity=person_purchase_amount_sensitivity))

    return people


def create_portfolio():
    offer_channel = Categorical(('web', 'email', 'mobile', 'social'), (1, 1, 1, 0))
    offer_type = Categorical(('bogo', 'discount', 'informational'), (1, 0, 0))
    bogo_a = Offer(0, valid_from=0, valid_until=1*7*24, difficulty=5, reward=5, channel=offer_channel, offer_type=offer_type)

    offer_channel = Categorical(('web', 'email', 'mobile', 'social'), (1, 1, 1, 1))
    offer_type = Categorical(('bogo', 'discount', 'informational'), (1, 0, 0))
    bogo_b = Offer(0, valid_from=0, valid_until=1*7*24, difficulty=10, reward=10, channel=offer_channel, offer_type=offer_type)

    offer_channel = Categorical(('web', 'email', 'mobile', 'social'), (1, 1, 1, 1))
    offer_type = Categorical(('bogo', 'discount', 'informational'), (0, 1, 0))
    discount_c = Offer(0, valid_from=0, valid_until=1*7*24, difficulty=10, reward=2, channel=offer_channel, offer_type=offer_type)

    offer_channel = Categorical(('web', 'email', 'mobile', 'social'), (1, 1, 1, 1))
    offer_type = Categorical(('bogo', 'discount', 'informational'), (0, 1, 0))
    discount_d = Offer(0, valid_from=0, valid_until=1*7*24, difficulty=7, reward=2, channel=offer_channel, offer_type=offer_type)

    offer_channel = Categorical(('web', 'email', 'mobile', 'social'), (1, 1, 1, 1))
    offer_type = Categorical(('bogo', 'discount', 'informational'), (0, 0, 1))
    info_e = Offer(0, valid_from=0, valid_until=4*7*24, difficulty=0, reward=0, channel=offer_channel, offer_type=offer_type)

    portfolio = (bogo_a, bogo_b, discount_c, discount_d, info_e)

    return portfolio


def assign_offers_to_subpopulation(population, subpopulation, deliveries_file_name, deliveries_log_file_name, control_fraction=0.25, delimiter='|', clean_path=True):

    if clean_path:
        data_file_names = glob.glob(os.path.join(population.deliveries_path, '*'))
        for data_file_name in data_file_names:
            os.remove(data_file_name)

    offer_ids = population.portfolio.keys()

    # update the validity dates of the offers
    now = population.world.world_time
    for offer in population.portfolio.values():
        offer_length = offer.valid_until - offer.valid_from
        offer.valid_from = now
        offer.valid_until = now + offer_length

    # make random delivery decisions
    deliveries = list()
    for person_id in subpopulation:
        assert person_id in population.people, 'ERROR - an indiviual in the desired subpopulation is not part of the general population: {}'.format(person_id)
        # hold out a fraction of people as control
        if numpy.random.random() < 1.0 - control_fraction:
            # make random deliveries to the rest
            deliveries.append((person_id, numpy.random.choice(offer_ids)))

    # write the deliveries file
    with open(deliveries_file_name, 'w') as deliveries_file:
        for delivery in deliveries:
            print >> deliveries_file, delimiter.join(map(str, delivery))

    # make a copy of the delivery file
    shutil.copy(deliveries_file_name, deliveries_log_file_name)


def main(args):
    world = World(real_time_tick=0.000, world_time_tick=6)

    people = list()
    people.extend(create_people_0(200))
    people.extend(create_people_1(500))
    people.extend(create_people_2(300))

    portfolio = create_portfolio()

    delivery_log_path = os.path.join(args.data_path, 'delivery_log')
    deliveries_path = os.path.join(args.data_path, 'delivery')
    transcripts_file_name = os.path.join(args.data_path, 'transcript.json')
    population_file_name = os.path.join(args.data_path, 'population.json')

    # clean up from previous runs if there's data present
    old_files = glob.glob(os.path.join(args.data_path, '*'))

    for file_name in old_files:
        if os.path.isfile(file_name):
            os.remove(file_name)

    # create directories if they don't already exist
    mkdir_if_missing(args.data_path)
    mkdir_if_missing(delivery_log_path)
    mkdir_if_missing(deliveries_path)

    # initialize
    population = Population(world,
                            people=people,
                            portfolio=portfolio,
                            deliveries_path=deliveries_path,
                            transcript_file_name=transcripts_file_name)

    with open(population_file_name, 'w') as population_file:
        print >> population_file, population.to_json()


    # main simulatio loop
    has_received_decision = set()
    sample_size_per_week = int(len(population.people) / 4.0)

    for week in range(4):
        print 'Starting week {}'.format(week)

        deliveries_file_short_name = 'deliveries.week_{}.csv'.format(week)
        deliveries_file_name = os.path.join(deliveries_path, deliveries_file_short_name)
        deliveries_log_file_name = os.path.join(delivery_log_path, deliveries_file_short_name)

        subpop_remaining = set(population.people.keys()) - has_received_decision
        subpop = numpy.random.choice(list(subpop_remaining), max(sample_size_per_week, len(subpop_remaining)))

        # make deliveries
        assign_offers_to_subpopulation(population, subpop, deliveries_file_name, deliveries_log_file_name)

        has_received_decision = has_received_decision.union(set(subpop))

        # update world by one weelk
        population.simulate(n_ticks=4*7, n_proc=args.n_proc)

    return 0


def get_args():
    """Build arg parser and get command line arguments

    :return: parsed args namespace
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", default="data",           help="data path file name")
    parser.add_argument("--n-proc",    default=1,      type=int, help="number of Processes to use for simulation")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main(get_args())

