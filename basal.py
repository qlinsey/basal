import torch
import torch.nn as nn
import torch.optim as optim

import os
import numpy as np
from sklearn.metrics import accuracy_score

import sampler
import copy
import time

from gutils import *
import torch.nn.functional as F


class Basaler:
    def __init__(self, args, device, label_list):
        self.args = args
        self.device = device
        self.label_list = label_list

    def train(
        self,
        train_dataloader,
        unlabeled_dataloader,
        test_dataloader,
        transformer,
        generator,
        discriminator,
        discriminator3,
    ):

        num_train_epochs = self.args.train_epochs

        training_stats = []

        # Measure the total training time for the whole run.
        total_t0 = time.time()

        # models parameters
        transformer_vars = [i for i in transformer.parameters()]
        d_vars = transformer_vars + [v for v in discriminator.parameters()]
        g_vars = [v for v in generator.parameters()]
        d3_vars = transformer_vars + [v for v in discriminator3.parameters()]

        # optimizer
        dis_optimizer = torch.optim.AdamW(
            d_vars, lr=self.args.learning_rate_discriminator
        )
        gen_optimizer = torch.optim.AdamW(g_vars, lr=self.args.learning_rate_generator)
        dis3_optimizer = torch.optim.AdamW(
            d3_vars, lr=self.args.learning_rate_discriminator
        )

        criterion = HLoss()
        bce_loss = nn.BCELoss()

        # scheduler
        if self.args.apply_scheduler:
            num_train_examples = len(train_examples)
            num_train_steps = int(num_train_examples / batch_size * num_train_epochs)
            num_warmup_steps = int(num_train_steps * warmup_proportion)

            scheduler_d = get_constant_schedule_with_warmup(
                dis_optimizer, num_warmup_steps=num_warmup_steps
            )
            scheduler_g = get_constant_schedule_with_warmup(
                gen_optimizer, num_warmup_steps=num_warmup_steps
            )

            scheduler_d3 = get_constant_schedule_with_warmup(
                dis3_optimizer, num_warmup_steps=num_warmup_steps
            )

        # For each epoch...
        for epoch_i in range(0, num_train_epochs):
            # ========================================
            #               Training
            # ========================================

            print("")
            print(
                "======== Epoch {:} / {:} ========".format(
                    epoch_i + 1, num_train_epochs
                )
            )
            print("Training...")

            t0 = time.time()
            tr_g_loss = 0
            tr_d_loss = 0
            # add tr_d2_loss and add one more term in tr_g_Loss
            u_tr_d_loss = 0

            #  training mode.
            transformer.train()
            generator.train()
            discriminator.train()
            # discriminator2.train()
            discriminator3.train()

            for step, batch in enumerate(zip(train_dataloader, unlabeled_dataloader)):

                # Progress update
                if step % self.args.print_each_n_step == 0 and not step == 0:
                    # Calculate elapsed time
                    elapsed = format_time(time.time() - t0)

                    # Report progress.
                    print(
                        "  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.".format(
                            step, len(train_dataloader), elapsed
                        )
                    )

                real_batch_size = min(batch[0][0].shape[0], batch[1][0].shape[0])

                b_input_ids = batch[0][0][:real_batch_size].to(self.device)
                b_input_mask = batch[0][1][:real_batch_size].to(self.device)
                b_labels = batch[0][2][:real_batch_size].to(self.device)
                b_label_mask = batch[0][3][:real_batch_size].to(self.device)

                u_input_ids = batch[1][0][:real_batch_size].to(self.device)
                u_input_mask = batch[1][1][:real_batch_size].to(self.device)
                u_labels = batch[1][2][:real_batch_size].to(self.device)
                u_label_mask = batch[1][3][:real_batch_size].to(self.device)

                # Encode real data in the Transformer
                model_outputs = transformer(b_input_ids, attention_mask=b_input_mask)
                hidden_states = model_outputs[-1]

                # noisy input are used in input to the Generator
                noise = torch.zeros(
                    real_batch_size, self.args.noise_size, device=self.device
                ).uniform_(0, 1)
                # Gnerate Fake data
                gen_rep = generator(noise)

                # Generate the output of the Discriminator for real and fake data.
                disciminator_input = torch.cat([hidden_states, gen_rep], dim=0)
                # output of the disciminator
                features, logits, probs = discriminator(disciminator_input)

                _, logits3, probs3 = discriminator3(hidden_states)
                lab_real_preds = torch.ones(len(probs3))

                # the real and fake
                features_list = torch.split(features, real_batch_size)
                D_real_features = features_list[0]
                D_fake_features = features_list[1]

                logits_list = torch.split(logits, real_batch_size)
                D_real_logits = logits_list[0]
                D_fake_logits = logits_list[1]

                probs_list = torch.split(probs, real_batch_size)
                D_real_probs = probs_list[0]
                D_fake_probs = probs_list[1]

                # ---------------------------------
                #  LOSS evaluation
                # ---------------------------------
                # Generator's LOSS estimation
                g_loss_d = -1 * torch.mean(
                    torch.log(1 - D_fake_probs[:, -1] + self.args.epsilon)
                )
                g_feat_reg = torch.mean(
                    torch.pow(
                        torch.mean(D_real_features, dim=0)
                        - torch.mean(D_fake_features, dim=0),
                        2,
                    )
                )
                g_loss = g_loss_d + g_feat_reg

                # ---------------------------------
                #  LOSS evaluation - Unlabel vs. Real Generation Loss
                # ---------------------------------
                # Generator's LOSS estimation

                #  Encode unlabelled data in the Transformer
                u_model_outputs = transformer(u_input_ids, attention_mask=u_input_mask)
                u_hidden_states = u_model_outputs[-1]
                u_disciminator_input = torch.cat([u_hidden_states, gen_rep], dim=0)
                # Then, we select the output of the disciminator
                u_features, u_logits, u_probs = discriminator(u_disciminator_input)

                _, u_logits3, u_probs3 = discriminator3(u_hidden_states)

                unlab_real_preds = torch.ones(len(u_probs3))

                u_features_list = torch.split(u_features, real_batch_size)
                u_D_real_features = u_features_list[0]
                u_D_fake_features = u_features_list[1]

                u_logits_list = torch.split(u_logits, real_batch_size)
                u_D_real_logits = u_logits_list[0]
                u_D_fake_logits = u_logits_list[1]

                u_probs_list = torch.split(u_probs, real_batch_size)
                u_D_real_probs = u_probs_list[0]
                u_D_fake_probs = u_probs_list[1]

                u_g_loss_d = -1 * torch.mean(
                    torch.log(1 - u_D_fake_probs[:, -1] + self.args.epsilon)
                )
                u_g_feat_reg = torch.mean(
                    torch.pow(
                        torch.mean(u_D_real_features, dim=0)
                        - torch.mean(u_D_fake_features, dim=0),
                        2,
                    )
                )
                u_g_loss = u_g_loss_d + u_g_feat_reg

                u_logits = u_D_real_logits[:, 0:-1]
                u_log_probs = F.log_softmax(u_logits, dim=-1)
                entropy_loss = criterion(u_logits)

                totoal_g_loss = g_loss + u_g_loss  # - entropy_loss

                # Disciminator's LOSS estimation - Disciminator1
                logits = D_real_logits[:, 0:-1]
                log_probs = F.log_softmax(logits, dim=-1)

                label2one_hot = torch.nn.functional.one_hot(
                    b_labels, len(self.label_list)
                )
                # print("label2one_hot.shape=",label2one_hot.shape,"log_probs.shape=",log_probs.shape)
                per_example_loss = -torch.sum(label2one_hot * log_probs, dim=-1)
                per_example_loss = torch.masked_select(
                    per_example_loss, b_label_mask.to(self.device)
                )
                labeled_example_count = per_example_loss.type(torch.float32).numel()

                if labeled_example_count == 0:
                    D_L_Supervised = 0
                else:
                    D_L_Supervised = torch.div(
                        torch.sum(per_example_loss.to(self.device)),
                        labeled_example_count,
                    )

                D_L_unsupervised1U = -1 * torch.mean(
                    torch.log(1 - D_real_probs[:, -1] + self.args.epsilon)
                )
                D_L_unsupervised2U = -1 * torch.mean(
                    torch.log(D_fake_probs[:, -1] + self.args.epsilon)
                )

                dsc_loss = bce_loss(
                    probs3[:, 0], lab_real_preds.to(self.device)
                ) + bce_loss(u_probs3[:, 0], unlab_real_preds.to(self.device))
                dsc_loss = dsc_loss

                u_D_L_unsupervised1U = -1 * torch.mean(
                    torch.log(1 - u_D_real_probs[:, -1] + self.args.epsilon)
                )
                u_D_L_unsupervised2U = -1 * torch.mean(
                    torch.log(u_D_fake_probs[:, -1] + self.args.epsilon)
                )

                d_loss = D_L_Supervised + u_D_L_unsupervised1U + u_D_L_unsupervised2U

                total_d_loss = d_loss  # - entropy_loss

                # ---------------------------------
                #  OPTIMIZATION
                # ---------------------------------
                gen_optimizer.zero_grad()
                dis_optimizer.zero_grad()
                dis3_optimizer.zero_grad()

                totoal_g_loss.backward(retain_graph=True)
                total_d_loss.backward(retain_graph=True)
                dsc_loss.backward()

                # update
                gen_optimizer.step()
                dis_optimizer.step()
                dis3_optimizer.step()

                # Save the losses to print them later
                tr_g_loss += totoal_g_loss.item()
                tr_d_loss += d_loss.item()
                # u_tr_d_loss += u_d_loss.item()

                # Update the learning rate with the scheduler
                if self.args.apply_scheduler:
                    scheduler_d.step()
                    scheduler_g.step()
                    scheduler_d3.step()

            # Calculate the average loss over all of the batches.
            avg_train_loss_g = tr_g_loss / len(train_dataloader)
            avg_train_loss_d = tr_d_loss / len(train_dataloader)
            # avg_u_train_loss_d = u_tr_d_loss / len(unlabeled_dataloader)

            # Measure how long this epoch took.
            training_time = format_time(time.time() - t0)

            print("")
            print("  Average training loss generetor: {0:.3f}".format(avg_train_loss_g))
            print(
                "  Average training loss discriminator: {0:.3f}".format(
                    avg_train_loss_d
                )
            )
            # print("  Average training u_loss discriminator: {0:.3f}".format(avg_u_train_loss_d))
            print("  Training epcoh took: {:}".format(training_time))

            # ========================================
            #     TEST ON THE EVALUATION DATASET
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our test set.
            print("")
            print("Running Test...")

            t0 = time.time()

            # valuation mode-
            transformer.eval()
            discriminator.eval()
            discriminator3.eval()
            generator.eval()

            # Tracking variables
            total_test_accuracy = 0

            total_test_loss = 0
            nb_test_steps = 0

            all_preds = []
            all_labels_ids = []

            # loss
            nll_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

            # Evaluate
            for batch in test_dataloader:

                # Unpack this training batch from our dataloader.
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                with torch.no_grad():
                    model_outputs = transformer(
                        b_input_ids, attention_mask=b_input_mask
                    )
                    hidden_states = model_outputs[-1]
                    _, logits, probs = discriminator(hidden_states)
                    ###log_probs = F.log_softmax(probs[:,1:], dim=-1)
                    filtered_logits = logits[:, 0:-1]
                    # Accumulate the test loss.
                    total_test_loss += nll_loss(filtered_logits, b_labels)

                # Accumulate the predictions and the input labels
                _, preds = torch.max(filtered_logits, 1)
                all_preds += preds.detach().cpu()
                all_labels_ids += b_labels.detach().cpu()

            #  final accuracy for this validation run.
            all_preds = torch.stack(all_preds).numpy()
            all_labels_ids = torch.stack(all_labels_ids).numpy()
            test_accuracy = np.sum(all_preds == all_labels_ids) / len(all_preds)
            print("  Accuracy: {0:.3f}".format(test_accuracy))

            # Calculate the average loss
            avg_test_loss = total_test_loss / len(test_dataloader)
            avg_test_loss = avg_test_loss.item()

            test_time = format_time(time.time() - t0)

            print("  Test Loss: {0:.3f}".format(avg_test_loss))
            print("  Test took: {:}".format(test_time))

            # output all statistics
            training_stats.append(
                {
                    "epoch": epoch_i + 1,
                    "Training Loss generator": avg_train_loss_g,
                    "Training Loss discriminator": avg_train_loss_d,
                    # 'Training U-Loss discriminator': avg_u_train_loss_d,
                    "Valid. Loss": avg_test_loss,
                    "Valid. Accur.": test_accuracy,
                    "Training Time": training_time,
                    "Test Time": test_time,
                }
            )

        torch.save(
            training_stats, os.path.join(args.train_out_path, args.train_log_name)
        )
        return test_accuracy, generator, transformer, discriminator, discriminator3

    def sample_for_labeling(self, bertmodel, discriminator, unlabeled_dataloader):
        unlabel_preds = []
        un_labels_ids = []
        pre_label_probs = []
        pre_labels = []
        k = 10
        real_batch = 20  # 52
        cnt = 0
        transformer = AutoModel.from_pretrained(bertmodel)
        if not discriminator:
            # discrimitor= Discriminator(input_size=self.args.hidden_size, hidden_sizes=self.args.hidden_levels_d, num_labels=len(label_list), dropout_rate=self.args.out_dropout_rate)
            discrimitor.load_state_dict(torch.load("./ganbart-discrminator2.pt"))
            discrimitor.eval()
        for cnt, batch in enumerate(unlabeled_dataloader):  # ):
            # print("cnt=",cnt,"batch[0].shape=",batch[0].shape[0])
            if batch[0].shape[0] != 32:
                continue
            t_input_ids = batch[0][0:real_batch].to(self.device)
            t_input_mask = batch[1][0:real_batch].to(self.device)
            t_labels = batch[2][0:real_batch].to(self.device)
            with torch.no_grad():
                model_outputs = transformer(t_input_ids, attention_mask=t_input_mask)
                hidden_states = model_outputs[-1]
                _, logits, probs = discriminator(hidden_states)
                ###log_probs = F.log_softmax(probs[:,1:], dim=-1)
                filtered_logits = logits[:, 0:-1]
                # Accumulate the test loss.
                total_test_loss += nll_loss(filtered_logits, t_labels)

            _, preds = torch.max(filtered_logits, 1)
            pre_label_idx = torch.argmax(F.softmax(filtered_logits, dim=-1), dim=1)
            pre_label_prob, _ = torch.max(F.softmax(filtered_logits, dim=-1), dim=1)
            # print("len(pre_label_prob)=",len(pre_label_prob), "len(pre_label_idx)=",len(pre_label_idx))
            # pre_prob = [pre_label_prob[i] for i in pre_label_idx.detach().cpu().numpy()]
            pre_label_probs += pre_label_prob
            pre_label = [label_list[i] for i in pre_label_idx.detach().cpu().numpy()]
            pre_labels += pre_label
            unlabel_preds += preds.detach().cpu()
            un_labels_ids += t_labels.detach().cpu()

        # print(unlabel_preds)
        # Report the final accuracy for this validation run.
        unlabel_preds_c = torch.stack(unlabel_preds).numpy()
        un_labels_ids = torch.stack(un_labels_ids).numpy()
        test_accuracy = np.sum(unlabel_preds_c == un_labels_ids) / len(unlabel_preds)
        print("  Accuracy: {0:.3f}".format(test_accuracy))
        print("unlabel_preds_c=", unlabel_preds_c[0:10])
        print("un_labels_ids=", un_labels_ids[0:10])

        pre_label_probs *= -1
        topk_ids = np.argsort(pre_label_probs)[-k:]
        topk_prob = [pre_label_probs[i].detach().cpu().numpy() for i in topk_ids]
        topk_pre_labels = un_labels_ids[topk_ids]
        topk_data_point = [unlabeled_examples[i] for i in topk_ids]

        print("topk_ids=", topk_ids)
        # querry_pool_indice

        return topk_ids
