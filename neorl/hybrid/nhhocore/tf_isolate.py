#    This file is part of NEORL.

#    Copyright (c) 2021 Exelon Corporation and MIT Nuclear Science and Engineering
#    NEORL is free software: you can redistribute it and/or modify
#    it under the terms of the MIT LICENSE

#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#    SOFTWARE.

import multiprocessing


def _run_and_capture(func, args, kwargs, result_queue):
    try:
        result = func(*args, **kwargs)
    except BaseException as exc:
        result_queue.put(('error', exc))
    else:
        result_queue.put(('ok', result))


def run_isolated(func, *args, **kwargs):
    """
    Runs func(*args, **kwargs) in a freshly spawned child process so Keras/TF
    starts with a clean eager-mode runtime, regardless of whether
    tf.compat.v1.disable_v2_behavior() was already toggled elsewhere in this
    process (e.g. by neorl's TF1 graph-mode RL baselines: A2C/ACER/DQN/PPO2/ACKTR).
    Using 'fork' would inherit that already-poisoned state, so 'spawn' is required.
    func, its arguments, and its return value must all be picklable.
    """
    # Inside a joblib(loky) worker, multiprocessing.get_start_method() reports
    # 'loky' instead of the real method, which corrupts the preparation data
    # for this nested spawn and crashes the child with
    # "ValueError: cannot find context for 'loky'". Force it back to 'spawn'
    # before creating the context so the nested child boots correctly.
    multiprocessing.set_start_method('spawn', force=True)
    ctx = multiprocessing.get_context('spawn')
    result_queue = ctx.Queue()
    process = ctx.Process(target=_run_and_capture, args=(func, args, kwargs, result_queue))
    process.start()
    status, payload = result_queue.get()
    process.join()
    if status == 'error':
        raise payload
    return payload
