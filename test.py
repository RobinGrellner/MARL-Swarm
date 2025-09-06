from envs.evader.test_evader_environment import TestEvaderEnvironment
import unittest

def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestEvaderEnvironment('test_pettingzoo_tests'))
    suite.addTest(TestEvaderEnvironment('test_agent_num'))
    suite.addTest(TestEvaderEnvironment('test_init'))
    suite.addTest(TestEvaderEnvironment('test_agent_movement_simple'))
    suite.addTest(TestEvaderEnvironment('test_agent_movement_rotation_only'))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())