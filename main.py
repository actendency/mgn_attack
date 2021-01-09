import data
import loss
import torch
import model
from trainer import Trainer
import numpy as np
from option import args
import utils.utility as utility
import torch.nn.functional as F
ckpt = utility.checkpoint(args)

loader = data.Data(args)
model = model.Model(args, ckpt)
loss = loss.Loss(args, ckpt)
trainer = Trainer(args, model, loss, loader, ckpt)

def save_img(adversarial_examples,l):
	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]
	for (id) in range(l):
		image_numpy = adversarial_examples[id].cpu().float().detach().numpy()
		for i in range(len(mean)):
			image_numpy[i] = image_numpy[i] * std[i] + mean[i]
		image_numpy = image_numpy * 255
		image_numpy = np.transpose(image_numpy, (1, 2, 0))
		image_numpy = image_numpy.astype(np.uint8)
		import imageio
		imageio.imwrite('adv/' + str(id) + '.jpg', image_numpy)

def attack1():

	attack_loader = loader.attack_loader
	for (i, (inputs, labels)) in enumerate(attack_loader):
		print(i)
		inputs.requires_grad = True
		y = model(inputs)
		tt = torch.norm(y[0][0] - y[0][0])
		for (id) in range(6):
			for (idd) in range(id):
				if (id != idd):
					tt = tt + torch.norm(y[0][i] - y[0][idd])
		# tt = y[0][0] - y[0][1]
		print(tt)
		print("11111111111111111111111")
		gradients = torch.autograd.grad(tt, inputs)[0]
		print(gradients.shape)
		adversarial_examples = inputs + (0.1 * gradients.sign())
		g = model(adversarial_examples)
		uu = torch.norm(g[0][0] - g[0][0])
		for (id) in range(inputs.shape[0]):
			for (idd) in range(id):
				if (id != idd):
					uu = uu + torch.norm(g[0][i] - g[0][idd])
		print(uu)
		save_img(adversarial_examples,inputs.shape[0])

def attack2():
	attack_loader = loader.attack_loader
	loss.start_log()
	for batch, (inputs, labels) in enumerate(attack_loader):
		inputs = inputs.to('cuda')
		inputs.requires_grad = True
		labels = labels.to('cuda')

		outputs = model(inputs)

		print(inputs.shape)
		print(labels.shape)
		print(outputs[0].shape)
		print("22222222222222222222222")

		lossx = loss(outputs, labels)
		#loss.backward()
		gradients = torch.autograd.grad(lossx, inputs)[0]
		adversarial_examples = inputs + (0.1 * gradients.sign())
		save_img(adversarial_examples, inputs.shape[0])
		loss.end_log(len(attack_loader))


def attack3():
	attack_loader = loader.attack_loader
	loss.start_log()
	for batch, (inputs, labels) in enumerate(attack_loader):
		inputs = inputs.to('cuda')
		inputs.requires_grad = True
		labels = labels.to('cuda')
		print("333333333333333333")
		outputs = model(inputs)
		lossx = loss(outputs, labels)
		gradients = torch.autograd.grad(lossx, inputs)[0]
		per = 0.1 * gradients.sign()
		for(t) in range(14):
			adversarial_examples = inputs + per
			#adversarial_examples = torch.clamp(adversarial_examples, min=0, max=1).detach()
			#adversarial_examples.requires_grad = True
			outputs = model(adversarial_examples)
			lossx = loss(outputs, labels)
			gradients = torch.autograd.grad(lossx, adversarial_examples)[0]
			per = per + (0.1/14)* gradients.sign()
			per = torch.clamp(per,min=-0.1,max=0.1)
		#loss.backward()
		save_img(adversarial_examples, inputs.shape[0])
		loss.end_log(len(attack_loader))

def run():
	n = 0
	while not trainer.terminate():
		n += 1
		trainer.train()
		if args.test_every != 0 and n % args.test_every == 0:
			trainer.test()
#attack1()
#attack2()
#attack3()
#run()
# print(gradients.shape)
# print(norm)

#trainer.train()
#trainer.test()


