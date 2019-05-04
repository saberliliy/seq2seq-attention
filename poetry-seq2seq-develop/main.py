#! /usr/bin/env python
# -*- coding:utf-8 -*-

from plan import Planner
from predict import Seq2SeqPredictor
import sys
from imp import reload
import tensorflow as tf
tf.app.flags.DEFINE_boolean('cangtou', False, 'Generate Acrostic Poem')

reload(sys)
# sys.setdefaultencoding('utf8')

def get_cangtou_keywords(input):
    assert(len(input) == 4)
    return [c for c in input]

def main(cangtou=False):
    planner = Planner()
    with Seq2SeqPredictor() as predictor:
        # Run loop
        terminate = False
        while not terminate:
            try:
                inputs =input('Input Text:\n').strip()

                if not inputs:
                    print( 'Input cannot be empty!')
                elif inputs.lower() in ['quit', 'exit']:
                    terminate = True
                else:
                    if cangtou:
                        keywords = get_cangtou_keywords(inputs)
                    else:
                        # Generate keywords
                        keywords = planner.plan(inputs)
                        print(keywords)
                    # Generate poem
                    lines = predictor.predict(keywords)

                    # Print keywords and poem
                    print( 'Keyword:\t\tPoem:')
                    for line_number in range(4):
                        punctuation = u'，' if line_number % 2 == 0 else u'。'
                        print (u'{keyword}\t\t{line}{punctuation}'.format(
                            keyword=keywords[line_number],
                            line=lines[line_number],
                            punctuation=punctuation
                        ))

            except EOFError:
                terminate = True
            except KeyboardInterrupt:
                terminate = True
    print ('\nTerminated.')


if __name__ == '__main__':
    main(cangtou=tf.app.flags.FLAGS.cangtou)