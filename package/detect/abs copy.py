
import pickle, gzip, numpy as np, os, sys
np.set_printoptions(precision=2, linewidth=200, threshold=10000)
import keras
from keras.models import Model, Sequential, load_model, model_from_yaml
from keras import backend as K
import json
from preprocess import CIFAR10
import tensorflow as tf, imageio
np.random.seed(333)
tf.set_random_seed(333)
with open('config.json') as (config_file):
    config = json.load(config_file)
os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu_id']
w = config['w']
h = config['h']
num_classes = config['num_classes']
use_mask = True
count_mask = True
tdname = 'temp'
cifar = CIFAR10()
print ('mean', cifar.mean, 'std', cifar.std)
l_bounds = np.asarray([(0 - cifar.mean[0]) / cifar.std[0], (0 - cifar.mean[1]) / cifar.std[1], (0 - cifar.mean[2]) / cifar.std[2]])
h_bounds = np.asarray([(255 - cifar.mean[0]) / cifar.std[0], (255 - cifar.mean[1]) / cifar.std[1], (255 - cifar.mean[2]) / cifar.std[2]])
l_bounds = np.asarray([ l_bounds for _ in range(w * h) ]).reshape((1, w, h, 3))
h_bounds = np.asarray([ h_bounds for _ in range(w * h) ]).reshape((1, w, h, 3))

def getlayer_output(l_in, l_out, x, model):
    get_k_layer_output = K.function([model.layers[l_in].input, 0], [model.layers[l_out].output])
    return get_k_layer_output([x])[0]


def check_values(images, labels, model):
    maxes = {}
    for hl_idx in range(len(model.layers) - 1):
        if 'pool' in model.layers[hl_idx].name or 'batch' in model.layers[hl_idx].name or 'activation' in model.layers[hl_idx].name or 'dropout' in model.layers[hl_idx].name or 'predictions' in model.layers[hl_idx].name:
            continue
        n_neurons = model.layers[hl_idx].output_shape[(-1)]
        h_layer = getlayer_output(0, hl_idx, images, model).copy()
        key = ('{0}').format(model.layers[hl_idx].name)
        if key in maxes.keys():
            maxes[key].append(np.amax(h_layer))
        else:
            maxes[key] = [
             np.amax(h_layer)]

    return maxes


def sample_neuron(images, labels, model, mvs):
    all_ps = {}
    samp_k = config['samp_k']
    same_range = config['same_range']
    n_samples = config['n_samples']
    batch_size = config['samp_batch_size']
    n_images = images.shape[0]
    for hl_idx in range(0, len(model.layers) - 1):
        if 'pool' in model.layers[hl_idx].name or 'batch' in model.layers[hl_idx].name or 'activation' in model.layers[hl_idx].name or 'dropout' in model.layers[hl_idx].name or 'predictions' in model.layers[hl_idx].name or 'conv2d_9' == model.layers[hl_idx].name or 'flatten_1' == model.layers[hl_idx].name:
            continue
        n_neurons = model.layers[hl_idx].output_shape[(-1)]
        if same_range:
            vs = np.asarray([ i * samp_k for i in range(n_samples) ])
        else:
            tr = samp_k * max(mvs[model.layers[hl_idx].name]) / n_samples
            vs = np.asarray([ i * tr for i in range(n_samples) ])
        h_layer = getlayer_output(0, hl_idx, images, model).copy()
        nbatches = n_neurons // batch_size
        for nt in range(nbatches):
            l_h_t = []
            for neuron in range(batch_size):
                if len(h_layer.shape) == 4:
                    h_t = np.tile(h_layer, (n_samples, 1, 1, 1))
                else:
                    h_t = np.tile(h_layer, (n_samples, 1))
                for i, v in enumerate(vs):
                    if len(h_layer.shape) == 4:
                        h_t[i * n_images:(i + 1) * n_images, :, :, neuron + nt * batch_size] = v
                    else:
                        h_t[i * n_iamegs:(i + 1) * n_iamges, neuron + nt * batch_size] = v

                l_h_t.append(h_t)

            f_h_t = np.concatenate(l_h_t, axis=0)
            fps = getlayer_output(hl_idx + 1, len(model.layers) - 2, f_h_t, model)
            for neuron in range(batch_size):
                tps = fps[neuron * n_samples * n_images:(neuron + 1) * n_samples * n_images]
                for img_i in range(len(labels)):
                    img_name = ('{0}_0.jpg').format(labels[img_i])
                    ps_key = ('{0}_{1}_{2}').format(img_name, model.layers[hl_idx].name, neuron + nt * batch_size)
                    ps = [ tps[(img_i + n_images * i)] for i in range(n_samples) ]
                    ps = np.asarray(ps)
                    ps = ps.T
                    all_ps[ps_key] = np.copy(ps)

    return all_ps


def find_min_max(model_name, all_ps, cut_val=20, top_k=10):
    max_ps = {}
    max_vals = []
    n_classes = 0
    n_samples = 0
    for k in sorted(all_ps.keys()):
        all_ps[k] = all_ps[k][:, :cut_val]
        n_classes = all_ps[k].shape[0]
        n_samples = all_ps[k].shape[1]
        if 'predictions_cifa10' in k or 'flatten' in k or 'dropout' in k:
            continue
        vs = []
        for l in range(10):
            vs.append(np.amax(all_ps[k][l][all_ps[k].shape[1] // 5:]) - np.amin(all_ps[k][l][:all_ps[k].shape[1] // 5]))

        ml = np.argsort(np.asarray(vs))[(-1)]
        sml = np.argsort(np.asarray(vs))[(-2)]
        val = vs[ml] - vs[sml]
        max_vals.append(val)
        max_ps[k] = (ml, val)

    neuron_ks = []
    imgs = []
    for k in sorted(max_ps.keys()):
        nk = ('_').join(k.split('_')[2:])
        neuron_ks.append(nk)
        imgs.append(('_').join(k.split('_')[:2]))

    neuron_ks = list(set(neuron_ks))
    imgs = list(set(imgs))
    min_ps = {}
    min_vals = []
    for k in neuron_ks:
        vs = []
        ls = []
        vdict = {}
        for img in sorted(imgs):
            nk = img + '_' + k
            l = max_ps[nk][0]
            v = max_ps[nk][1]
            vs.append(v)
            ls.append(l)
            if l not in vdict.keys():
                vdict[l] = [
                 v]
            else:
                vdict[l].append(v)

        ml = max(set(ls), key=ls.count)
        tvs = []
        for img in sorted(imgs):
            nk = img + '_' + k
            l = max_ps[nk][0]
            v = max_ps[nk][1]
            tvs.append(v)

        fvs = []
        for img in sorted(imgs):
            img_l = int(img.split('_')[0])
            if img_l == ml:
                continue
            nk = img + '_' + k
            l = max_ps[nk][0]
            v = max_ps[nk][1]
            if l != ml:
                continue
            fvs.append(v)

        if len(fvs) == 0:
            for img in sorted(imgs):
                img_l = int(img.split('_')[0])
                if img_l == ml:
                    continue
                nk = img + '_' + k
                l = max_ps[nk][0]
                v = max_ps[nk][1]
                fvs.append(v)

        min_ps[k] = (
         l, ls.count(l), np.min(fvs), fvs)
        min_vals.append(np.min(fvs))

    keys = min_ps.keys()
    keys = []
    for k in min_ps.keys():
        if min_ps[k][1] >= n_samples - 2:
            keys.append(k)

    sorted_key = sorted(keys, key=lambda x: min_ps[x][2])
    neuron_dict = {}
    neuron_dict[model_name] = []
    maxval = min_ps[sorted_key[(-1)]][2]
    for i in range(min(len(sorted_key), top_k)):
        k = sorted_key[(-i - 1)]
        layer = k.split('_')[0] + '_' + k.split('_')[1]
        neuron = k.split('_')[(-1)]
        neuron_dict[model_name].append((layer, neuron, min_ps[k][0]))

    return neuron_dict


def read_all_ps(model_name, all_ps, top_k=10, cut_val=5):
    return find_min_max(model_name, all_ps, cut_val, top_k=top_k)


def filter_img(w, h):
    mask = np.zeros((h, w), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            if j >= 2 and j < 8 and i >= 2 and i < 8:
                mask[(i, j)] = 1

    return mask


def nc_filter_img(w, h):
    if use_mask:
        mask = np.zeros((h, w), dtype=np.float32)
        for i in range(h):
            for j in range(w):
                if not (j >= w * 1 / 4.0 and j < w * 3 / 4.0 and i >= h * 1 / 4.0 and i < h * 3 / 4.0):
                    mask[(i, j)] = 1

        mask = np.zeros((h, w), dtype=np.float32) + 1
    else:
        mask = np.zeros((h, w), dtype=np.float32) + 1
    return mask


def setup_model(optz_option, weights_file, Troj_Layer, Troj_next_Layer):
    nc_mask = nc_filter_img(w, h)
    with tf.variable_scope('', reuse=tf.AUTO_REUSE):
        mask = tf.get_variable('mask', [h, w], dtype=tf.float32)
        s_image = tf.placeholder(tf.float32, shape=(None, h, w, 3))
        if optz_option == 0:
            delta = tf.get_variable('delta', [1, h, w, 3], constraint=lambda x: tf.clip_by_value(x, l_bounds, h_bounds))
        elif optz_option == 1:
            delta = tf.placeholder(tf.float32, shape=(None, h, w, 3))
    con_mask = tf.tanh(mask) / 2.0 + 0.5
    con_mask = con_mask * nc_mask
    use_mask = tf.tile(tf.reshape(con_mask, (1, h, w, 1)), tf.constant([1, 1, 1, 3]))
    i_image = s_image * (1 - use_mask) + delta * use_mask
    model = load_model(str(weights_file))
    i_shape = model.get_layer(Troj_Layer).output_shape
    ni_shape = model.get_layer(Troj_next_Layer).output_shape
    t1_model = keras.models.clone_model(model)
    while t1_model.layers[(-1)].name != Troj_Layer:
        t1_model.pop()

    tinners = t1_model(i_image)
    t2_model = keras.models.clone_model(model)
    while t2_model.layers[(-1)].name != Troj_next_Layer:
        t2_model.pop()

    ntinners = t2_model(i_image)
    t3_model = keras.models.clone_model(model)
    t3_model.pop()
    logits = t3_model(i_image)
    models = [
     model, t1_model, t2_model, t3_model]
    return (
     models, i_image, s_image, delta, mask, con_mask, tinners, ntinners, i_shape, ni_shape, logits)


def define_graph(optz_option, Troj_Layer, Troj_Neuron, Troj_next_Layer, Troj_next_Neuron, variables1, Troj_size=64):
    models, i_image, s_image, delta, mask, con_mask, tinners, ntinners, i_shape, ni_shape, logits = variables1
    if len(i_shape) == 2:
        vloss1 = tf.reduce_sum(tinners[:, Troj_Neuron])
        vloss2 = 0
        if Troj_Neuron > 0:
            vloss2 += tf.reduce_sum(tinners[:, :Troj_Neuron])
        if Troj_Neuron < i_shape[(-1)] - 1:
            vloss2 += tf.reduce_sum(tinners[:, Troj_Neuron + 1:])
    elif len(i_shape) == 4:
        vloss1 = tf.reduce_sum(tinners[:, :, :, Troj_Neuron])
        vloss2 = 0
        if Troj_Neuron > 0:
            vloss2 += tf.reduce_sum(tinners[:, :, :, :Troj_Neuron])
        if Troj_Neuron < i_shape[(-1)] - 1:
            vloss2 += tf.reduce_sum(tinners[:, :, :, Troj_Neuron + 1:])
    if len(ni_shape) == 2:
        relu_loss1 = tf.reduce_sum(ntinners[:, Troj_next_Neuron])
        relu_loss2 = 0
        if Troj_Neuron > 0:
            relu_loss2 += tf.reduce_sum(ntinners[:, :Troj_next_Neuron])
        if Troj_Neuron < i_shape[(-1)] - 1:
            relu_loss2 += tf.reduce_sum(ntinners[:, Troj_next_Neuron + 1:])
    if len(ni_shape) == 4:
        relu_loss1 = tf.reduce_sum(ntinners[:, :, :, Troj_next_Neuron])
        relu_loss2 = 0
        if Troj_Neuron > 0:
            relu_loss2 += tf.reduce_sum(ntinners[:, :, :, :Troj_next_Neuron])
        if Troj_Neuron < i_shape[(-1)] - 1:
            relu_loss2 += tf.reduce_sum(ntinners[:, :, :, Troj_next_Neuron + 1:])
    tvloss = tf.reduce_sum(tf.image.total_variation(delta))
    loss = -vloss1 - relu_loss1 + 0.0001 * vloss2 + 0.0001 * relu_loss2
    mask_loss = tf.reduce_sum(con_mask)
    mask_cond1 = tf.greater(mask_loss, tf.constant(float(Troj_size)))
    mask_cond2 = tf.greater(mask_loss, tf.constant(100.0))
    mask_nz = tf.count_nonzero(tf.nn.relu(con_mask - 0.01), dtype=tf.int32)
    if count_mask:
        mask_cond1 = tf.greater(mask_nz, tf.constant(Troj_size))
        mask_cond2 = tf.greater(mask_nz, tf.constant(int((np.sqrt(Troj_size) + 2) ** 2)))
    loss += tf.cond(mask_cond1, true_fn=lambda : tf.cond(mask_cond2, true_fn=lambda : 1000 * mask_loss, false_fn=lambda : 500 * mask_loss), false_fn=lambda : 0.0 * mask_loss)
    lr = 0.2
    if use_mask:
        train_op = tf.train.AdamOptimizer(lr).minimize(loss, var_list=[delta, mask])
    else:
        train_op = tf.train.AdamOptimizer(lr).minimize(loss, var_list=[delta])
    grads = tf.gradients(loss, delta)
    return (
     models, s_image, tinners, logits, loss, vloss1, vloss2, tvloss, relu_loss1, relu_loss2, i_image, delta, mask, con_mask, train_op, grads, i_shape, ni_shape, mask_nz, mask_loss, mask_cond1)


def reverse_engineer(optz_option, images, weights_file, Troj_Layer, Troj_Neuron, Troj_next_Layer, Troj_next_Neuron, Troj_Label, variables2, RE_img='./adv.png', RE_delta='./delta.pkl', RE_mask='./mask.pkl', Troj_size=64):
    models, s_image, tinners, logits, loss, vloss1, vloss2, tvloss, relu_loss1, relu_loss2, i_image, delta, mask, con_mask, train_op, grads, i_shape, ni_shape, mask_nz, mask_loss, mask_cond1 = variables2
    Troj_Idx = 0
    Troj_next_Idx = 0
    n_weights = 0
    for i in range(len(models[0].layers)):
        n_weights += len(models[0].layers[i].get_weights())
        if models[0].layers[i].name == Troj_Layer:
            Troj_Idx = n_weights
        if models[0].layers[i].name == Troj_next_Layer:
            Troj_next_Idx = n_weights

    if use_mask:
        mask_init = filter_img(h, w) * 4 - 2
    else:
        mask_init = filter_img(h, w) * 8 - 4
    delta_init = np.random.normal(np.float32([0]), 1, (h, w, 3))
    delta_init = np.reshape(delta_init, (1, h, w, 3))
    with tf.Session() as (sess):
        init = tf.global_variables_initializer()
        sess.run(init)
        sess.run(delta.assign(delta_init))
        sess.run(mask.assign(mask_init))
        models[0].load_weights(weights_file)
        models[1].set_weights(models[0].get_weights()[:Troj_Idx])
        models[2].set_weights(models[0].get_weights()[:Troj_next_Idx])
        models[3].set_weights(models[0].get_weights())
        ot_loss = 0
        oo_loss = 0
        nt_loss = 0
        no_loss = 0
        K.set_learning_phase(0)
        if optz_option == 0:
            rlogits, rloss, rloss1, rloss2, rtvloss, rrelu_loss1, rrelu_loss2, adv, rdelta = sess.run((logits, loss, vloss1, vloss2, tvloss, relu_loss1, relu_loss2, i_image, delta), {s_image: images})
            ot_loss = rrelu_loss1
            oo_loss = rrelu_loss2
            for e in range(1000):
                rinner, rlogits, rloss, rloss1, rloss2, rtvloss, rrelu_loss1, rrelu_loss2, rmask_nz, rmask_cond1, rmask_loss, adv, rdelta, _ = sess.run((tinners, logits, loss, vloss1, vloss2, tvloss, relu_loss1, relu_loss2, mask_nz, mask_cond1, mask_loss, i_image, delta, train_op), {s_image: images})

            rlogits, rloss, rloss1, rloss2, rtvloss, rrelu_loss1, rrelu_loss2, adv, rdelta = sess.run((logits, loss, vloss1, vloss2, tvloss, relu_loss1, relu_loss2, i_image, delta), {s_image: images})
            nt_loss = rrelu_loss1
            no_loss = rrelu_loss2
        rlogits, rloss, rloss1, rloss2, rtvloss, rrelu_loss1, rrelu_loss2, rmask_loss, rcon_mask, rmask_nz, adv, rdelta = sess.run((logits, loss, vloss1, vloss2, tvloss, relu_loss1, relu_loss2, mask_loss, con_mask, mask_nz, i_image, delta), {s_image: images})
        adv = np.clip(adv, l_bounds, h_bounds)
        adv = cifar.deprocess(adv)
        adv = adv.astype('uint8')
        acc = np.sum(np.argmax(rlogits, axis=1) == Troj_Label) / float(rlogits.shape[0])
        return (
         acc, adv, rdelta, rcon_mask)


def re_mask(neuron_dict, layers, images):
    validated_results = []
    for key in sorted(neuron_dict.keys()):
        weights_file = key
        for task in neuron_dict[key]:
            Troj_Layer, Troj_Neuron, Troj_Label = task
            Troj_Neuron = int(Troj_Neuron)
            Troj_next_Layer = layers[(layers.index(Troj_Layer) + 1)]
            Troj_next_Neuron = Troj_Neuron
            Troj_size = config['troj_size']
            optz_option = 0
            RE_img = ('./imgs/{0}_model_{1}_{2}_{3}_{4}.png').format(weights_file.split('/')[(-1)][:-3], Troj_Layer, Troj_Neuron, Troj_size, Troj_Label)
            RE_mask = ('./masks/{0}_model_{1}_{2}_{3}_{4}.png').format(weights_file.split('/')[(-1)][:-3], Troj_Layer, Troj_Neuron, Troj_size, Troj_Label)
            RE_delta = ('./deltas/{0}_model_{1}_{2}_{3}_{4}.pkl').format(weights_file.split('/')[(-1)][:-3], Troj_Layer, Troj_Neuron, Troj_size, Troj_Label)
            variables1 = setup_model(optz_option, weights_file, Troj_Layer, Troj_next_Layer)
            variables2 = define_graph(optz_option, Troj_Layer, Troj_Neuron, Troj_next_Layer, Troj_next_Neuron, variables1, Troj_size)
            acc, rimg, rdelta, rmask = reverse_engineer(optz_option, images, weights_file, Troj_Layer, Troj_Neuron, Troj_next_Layer, Troj_next_Neuron, Troj_Label, variables2, RE_img, RE_delta, RE_mask, Troj_size)
            if acc >= 0.8:
                validated_results.append((rimg, rdelta, rmask, Troj_Label, RE_img, RE_mask, RE_delta))

        return validated_results


def stamp(n_img, delta, mask):
    mask0 = nc_filter_img(w, h)
    mask = mask * mask0
    r_img = n_img.copy()
    for i in range(h):
        for j in range(w):
            r_img[:, i, j] = n_img[:, i, j] * (1 - mask[(i, j)]) + delta[:, i, j] * mask[(i, j)]

    return r_img


def test_mask(weights_file, test_xs, result):
    rimg, rdelta, rmask, tlabel = result[:4]
    model = load_model(str(weights_file))
    func = K.function([model.input, K.learning_phase()], [model.get_layer('flatten_1').output])
    clean_images = cifar.preprocess(test_xs)
    t_images = stamp(clean_images, rdelta, rmask)
    for i in range(len(t_images)):
        imageio.imsave(tdname + '/' + ('{0}.png').format(i), cifar.deprocess(t_images[i]))

    nt_images = cifar.deprocess(t_images).astype('uint8')
    rt_images = cifar.preprocess(nt_images)
    yt = np.zeros(len(rt_images)) + tlabel
    yt = keras.utils.to_categorical(yt, num_classes)
    score = model.evaluate(rt_images, yt, verbose=0)
    return score[1]


if __name__ == '__main__':
    fxs, fys = pickle.load(open(config['img_pickle_file']))
    xs = fxs[:4]
    ys = fys[:4]
    test_xs = fxs
    test_ys = fys
    model = load_model(str(config['model_file']))
    model.summary()
    layers = [ l.name for l in model.layers ]
    processed_xs = cifar.preprocess(xs)
    maxes = check_values(processed_xs, ys, model)
    all_ps = sample_neuron(processed_xs, ys, model, maxes)
    neuron_dict = read_all_ps(config['model_file'], all_ps)
    print neuron_dict
    results = re_mask(neuron_dict, layers, processed_xs)
    reasrs = []
    for result in results:
        reasr = test_mask(str(config['model_file']), test_xs, result)
        reasrs.append(reasr)
        if reasr > 0.8:
            adv, rdelta, rmask, Troj_Label, RE_img, RE_mask, RE_delta = result
            for i in range(adv.shape[0]):
                imageio.imwrite(RE_img[:-4] + ('_{0}.png').format(i), adv[i])

            with open(RE_delta, 'wb') as (f):
                pickle.dump(rdelta, f)
            with open(RE_mask, 'wb') as (f):
                pickle.dump(rmask, f)

    print (
     str(config['model_file']), max(reasrs))