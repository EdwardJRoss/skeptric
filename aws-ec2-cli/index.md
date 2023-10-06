---
categories:
- aws
date: '2023-10-04T19:50:44+10:00'
image: aws_cli_reference.png
title: Starting AWS EC2 Compute Instances from the Command Line
format:
  html:
    toc: true
---

The ability to rent compute resources on demand is incredibly useful as a data scientist.
There are often batch processing jobs, such as training a machine learning model, that require large compute resources for a small period of time, and it's much more convenient to rent the resources as you need them rather than buying dedicated resources and trying to share them within a team.
Being able to quickly manage these through a REPL such as the command line makes it easy to quickly manage the resources you need.

Dealing with AWS it's helpful to have command line tools to interact with EC2 instances in flow.
Even though AWS isn't the cheapest compute cloud for GPUs (2-4x more expensive than things like [Lambda Labs](https://lambdalabs.com/service/gpu-cloud), [Jarvislabs.ai](https://jarvislabs.ai/pricing/) and similar), a lot of enterprises already have their sensitive data in AWS and so it often makes sense to pay the premium.
I'm going to concentrate on the small workflow where you want to run a process on a few machines directly, which gives you a lot of control and flexibility, but there are many higher levels of abstraction in AWS for different usecases:

* [Sagemaker](https://aws.amazon.com/sagemaker/) and [Databricks](https://www.databricks.com/) both provide a suite of machine learning tools centred around Notebook driven development
* [AWS Athena](https://aws.amazon.com/athena/) lets you do large data transformations with [Trino]( https://trino.io/) or [Apache Spark](https://spark.apache.org/) (or you could run your own on [AWS EMR](https://aws.amazon.com/emr/))
* You could run other distributed data transformation tools like [Dask](https://www.dask.org/) or [Ray](https://www.ray.io/)
* ML Data Orchestration tools like [Metaflow](https://metaflow.org/), [ZenML](https://www.zenml.io/integrations/aws), and [Kubeflow](https://www.kubeflow.org/) can be configured to run Python scripts on AWS
* [AWS Batch](https://aws.amazon.com/batch/) lets you run jobs on many EC2 instances (and is used by Metaflow)

For many simple usecases I find these tools are more complex to use or have unexpected edge cases, and find single EC2 instances simple enough, which I believe these other tools all build on top of.

There exist may command line tools for interacting with EC2 from the command-line, but they all fit specific use-cases and I want to build a thin wrapper around the AWS CLI.
The ones I have found are:

* The [AWS CLI](https://aws.amazon.com/cli/) has almost 600 ec2 commands which is very comprehensive but a bit overwhelming.
* [saws](https://github.com/donnemartin/saws) makes the AWS CLI easier to use with auto-completion
* [awless](https://github.com/wallix/awless) gives a nice interface for a subset of AWS services, but has limited configuration for EC2
* [simple-ec2](https://github.com/awslabs/aws-simple-ec2-cli), out of AWS labs, makes it easy to launch, connect and terminate to an instance, but doesn't allow you to do things like start, stop or list instances.
* [aec](https://github.com/seek-oss/aec) covers many EC2 features, but lacks documentation
* [fastec2](https://github.com/fastai/fastec2) has good basic features and a discoverable API, and is pretty close to what I need but misses a couple of configurations I need

For someone like me not an expert in AWS I'm not always sure what "sensible defaults" these tools choose, and they don't always fit into how I need to configure things in an organization.
I'm also a little wary of the security risk given tools to access servers would be a good target for a backdoor, like the Python [ssh-decorate module was](https://www.securitynewspaper.com/2018/05/10/researchers-found-backdoor-python-library-steal-ssh-credentials/).

some configuration options

My use case is very similar to fastec2, a way for regular people to run jobs on compute instances, and it's worth reading their articles about [using fastec2](https://www.fast.ai/posts/2019-02-19-fastec2.html) and [launching long running scripts](https://www.fast.ai/posts/2019-02-20-fastec2-script.html).
But here I'm going to just wrap the AWS CLI for some of the most common functions I need.
I'm going to assume you have the [AWS CLI](https://aws.amazon.com/cli/) installed and configured correctly.

# Basic Concepts

If you're not an expert in AWS EC2 here are a couple of key concepts it's useful to be familiar with; the [user guide](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/concepts.html) has more correct versions but here it is in words that I can understand:

* *instance* is a computer (actually it's a Virtual Machine running on a computer giving you access to some of its resources)
* *instance type* is the hardware on that computer (number of CPUs, amount of RAM, type of GPUs)
* *instance state* is what it is doing; it can be *running* (the computer is on), *stopped* (the computer is off) or *terminated* (the computer is destroyed)
* *EBS* (Elastic Block Store) is a hard drive you can attach to your instance, and can take *snapshots* (copies) of
* *AMI* (Amazon Machine Image, or just *image*) is like a hard drive containing the operating system (it's a type of Virtual Machine Image); Amazon provides many and you can create new ones from an existing EBS volume you've set up
* *key pair* is the cryptographic key that lets you access the machine remotely (you have the private key on your computer, the public key is on the server)
* *regions* and *availability zones* are about where the computers are physically located in data centres; some data centres have more computers available, and the further away they are from you the higher the latency will be to you
* *VPC* (Virtual Private Cloud) is about how the computers are networked together (in software), with a firewall managed by *Security groups*
* *Launch Template* is a bunch of defaults in specifying an instance, and can include all of the above

# List compute instances

It's often useful to know what instances are currently created and which of those are running.
This is done with the [`aws ec2 describe-instances`](https://docs.aws.amazon.com/cli/latest/reference/ec2/describe-instances.html) command.
When you run this you get a huge blob of JSON describing all the attributes of the image which can be a bit overwhelming.
However there's a `--query` argument that takes a [JMESPath](https://jmespath.org/) query to output just the information you want, and a `--output` field that can display it as `text` or a `table`.
For example a bash function to show the instance id, the value of the Name tag, the instance type, the state (whether it's running or shutdown), and the name of the encryption key you can define the bash function `ec2ls`:

```sh
ec2ls() {
  aws ec2 describe-instances --query "Reservations[].Instances[].{id: InstanceId, name: Tags[?Key=='Name'].Value | [0], type: InstanceType, state: State.Name, key: KeyName}" --output table $@
}
```

Note that for the `Name` tag we have to iterate over all the tag objects to find those containing name 'Key' with value 'Name', and pick the first one (and there may be issues if there is not exactly one name tag).
This produces output something like:

```
-----------------------------------------------------------------------
|                        DescribeInstances                            |
+----------------------+---------+---------+----------+---------------+
|          id          |   key   |  name   |  state   |     type      |
+----------------------+---------+---------+----------+---------------+
|  i-0123456789abcdef0 |  key-1  |  dev-1  |  running |  t2.micro     |
|  i-0123456789abcdef1 |  key-2  |  dev-2  |  stopped |  g4dn.xlarge  |
+----------------------+---------+---------+----------+---------------+
```

If you have a lot of instances you may also want to pass `--filters`, for example to only show running instances you can add `--filters Name=instance-state-name,Values=running`.

# Get an instance id by name

The EC2 instance ids are very hard to remember, so it's useful to have a human readable name to refer to them by.
When you create instances in the AWS Web User Interface (with "Click Ops") it prompts you for a human readable tag Name (which we displayed in `ec2ls` above).
We can use `describe-instances` to get the id of all objects with a given name and return them as text:

```sh
ec2id() {
  aws ec2 describe-instances --filter "Name=tag:Name,Values=$1" --query 'Reservations[].Instances[].InstanceId' --output text
}
```

Note that if there is more than 1 match they are all returned, and in particular you can use `*` for a wildcard match; so for example `ec2id dev-*` will return the ids for all instances with names starting with `dev-`.


# Start, stop, and terminate an instance

Now that we can get an instance by name we can easily start, stop, or terminate an instance by it's name with the appropriate commands.

```sh
ec2start() {
  aws ec2 start-instances --instance-ids $(ec2id $1)
}

ec2stop() {
  aws ec2 stop-instances --instance-ids $(ec2id $1)
}


ec2rm() {
  aws ec2 terminate-instances --instance-ids $(ec2id $1)
}
```

Note that these work even if there are multiple matches, so `ec2start dev-*` will start all the EC2 instances with names starting with `dev-`.
Also note that a terminated machine can't ever be used again, so in analogue to removing a file I call this `ec2rm`.

# Change an instance type

Sometimes you find you need a more or less powerful instance than you originally asked for; you can easily change the [type](https://aws.amazon.com/ec2/instance-types/) a stopped instance with [`aws ec2 modify-instance-attribute`](https://docs.aws.amazon.com/cli/latest/reference/ec2/modify-instance-attribute.html):

```sh
ec2mod() {
if [ "$#" -ne 2 ]; then
   echo "Usage: ec2mod <instance-name> <instance-type>"
   return 1
fi
aws ec2 modify-instance-attribute \
    --instance-id $(ec2id "$1") \
    --instance-type "$2"
}
```

For example to change all machines with names starting with `dev-` to `t2.micro` you can use `aws ec2 dev-* t2.micro`.

# Launch Instances

When you want to launch a new instance the [`aws ec2 run-instances`](https://docs.aws.amazon.com/cli/latest/reference/ec2/run-instances.html) command lets you create an instance.
Specifying an instance, and especially the attached EBS storage, is verbose to type at the command line, but AWS have the concept of a *launch template* that lets you specify defaults.
Launch templates can be [created](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/create-launch-template.html) either in the Web UI or with the AWS CLI, and then listed using `aws ec2 describe-launch-templates`.
We can then define a bash function `ec2mk` to make an instance tagged with a given Name from a specified launch template, with any other overrides passed in the command line:

```sh
ec2mk() {
if [ "$#" -lt 2 ]; then
   echo "Usage: ec2mod <instance-name> <launch-template> <*extra-args>"
   return 1
fi
instance_name=$1
launch_template=$2
shift 2

aws ec2 run-instances --dry-run \
   --launch-template LaunchTemplateName="$launch_template" \
   --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value="$instance_name"}]" \
   $@
}
```

# And more

You can extend this to do things like manage EBS Volumes, AMIs, Launch Templates, and more.
For example to list AMIs that I own (with a name containing the optional first argument), I would look through the CLI API for things containing `image`, find [`describe-images`](https://docs.aws.amazon.com/cli/latest/reference/ec2/describe-images.html) read the documentation, and write appropriate queries and filters:

```sh
ec2ami() {
aws ec2 describe-images \
    --executable-users self \
    --query "Images[].{Name: Name, ImageId: ImageId, CreationDate: CreationDate} | sort_by(@, &CreationDate) | reverse(@) " \
    --filter "Name=name,Values=*$1*" \
    --output table
}
```

You can put these in your `~/.bashrc` and use them to get started in simple cases, and extend them as you need.
If you end up spending a lot of time extending them you have enough understanding to start exploring some of the alternative CLIs and services listed at the start of the article.
It's worth noting if you want to automate in Python that [boto3 ec2](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ec2.html) has the same interface as the CLI, and you can execute commands on the servers over SSH with [fabric](https://www.fabfile.org/).
Have fun managing EC2 instances, but remember to `stop` them when you're finished or you'll get a nasty surprise bill!
