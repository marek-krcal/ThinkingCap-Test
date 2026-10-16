import unittest
from cloud.select_offline_training import select, task_score


class SelectionTests(unittest.TestCase):
    def row(self, id, text):
        return dict(id=id,dataset='skywork',seed_index=123,
                    messages=[dict(role='user',content=text)],split='train')

    def test_arithmetic_story_priority_over_abstract_math(self):
        story=self.row('s/1','Anna bought 4 books for 3 dollars each. How much money did she spend?')
        abstract=self.row('s/2',r'Solve $\log_2 x=3$ using a theorem.')
        self.assertGreater(task_score(story),task_score(abstract))
        self.assertEqual(select([abstract,story],{'skywork':1}),[story])

    def test_deterministic_identity_and_seed_preservation(self):
        rows=[self.row('s/'+str(i),'How many books are left?') for i in range(10)]
        chosen=select(rows,{'skywork':3})
        self.assertEqual(chosen,select(list(reversed(rows)),{'skywork':3}))
        self.assertTrue(all(r in rows and r['seed_index']==123 for r in chosen))
        with self.assertRaises(ValueError): select(rows,{'skywork':11})
