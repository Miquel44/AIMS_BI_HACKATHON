resource "aws_autoscaling_group" "app_asg" {
  name                      = "tf-asg"
  max_size                  = var.asg_max
  min_size                  = var.asg_min
  desired_capacity          = var.asg_min
  # Place instances in public subnets so they receive a public IP
  vpc_zone_identifier       = aws_subnet.public[*].id
  launch_template {
    id      = aws_launch_template.app_lt.id
    version = "${aws_launch_template.app_lt.latest_version}"
  }

  target_group_arns = [aws_lb_target_group.app_tg.arn]

  tag {
    key                 = "Name"
    value               = "tf-asg-instance"
    propagate_at_launch = true
  }

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_launch_template" "app_lt" {
  name_prefix   = "tf-app-lt-"
  image_id      = var.ami_id
  instance_type = var.instance_type
  key_name      = var.key_name != "" ? var.key_name : null
  iam_instance_profile {
    name = aws_iam_instance_profile.ec2_profile.name
  }

  network_interfaces {
    associate_public_ip_address = true
    security_groups             = [aws_security_group.instance_sg.id]
  }

  user_data = base64encode(var.user_data == "" ? <<-EOT
#!/bin/bash
set -e
# Amazon Linux 2023 compatible bootstrap: install required tools
sudo dnf -y update
# Install required packages including nginx so instances respond to ALB
sudo dnf -y install unzip jq git nginx

# Ensure nginx is enabled and started
sudo systemctl enable --now nginx

# 1. Crear el directorio de clonación
sudo mkdir -p /opt/app
sudo chown ec2-user:ec2-user /opt/app

# 2. Configurar .gitconfig para el usuario ec2-user
# Nota: Usamos 'sudo -u ec2-user' para crear el archivo en /home/ec2-user
sudo -u ec2-user tee /home/ec2-user/.gitconfig >/dev/null <<EOF
[user]
    name = opsora20
    email = sergiosmp20@gmail.com

[credential]
    helper = store

[core]
    autocrlf = true

# Para GitHub específicamente
[github]
    token = FILLIN_YOUR_GITHUB

# Reemplaza todas las peticiones a github.com con las credenciales incrustadas
[url "https://FILLIN_YOUR_GITHUB/"]
    insteadOf = https://github.com/
EOF

# 3. Clonar el repositorio usando la configuración de credenciales
# El comando 'git clone' como ec2-user usará el .gitconfig creado arriba.
echo "Clonando repositorio..."
sudo -u ec2-user git clone "https://github.com/Boehringer-hackathon/Equipo-equipo-aims-22.git" /opt/app

# 4. Limpieza (Opcional pero Recomendado para evitar que el token quede en el .git/config del repo)
# Ya que usamos insteadOf, el token no debería quedar, pero es buena práctica.
# La configuración del token está en .gitconfig del usuario, que es la forma correcta.

# 5. Despliegue del artefacto web
if [ -f /opt/app}/index.html ]; then
  sudo cp /opt/app/index.html /usr/share/nginx/html/index.html
fi

# 6. Asegurar que los archivos sean propiedad de ec2-user (Buena práctica)
sudo chown -R ec2-user:ec2-user /opt/app

# --- FIN: Configuración de Git ---

# 1. Descargar el instalador de Miniconda como ec2-user
# Usamos sudo -u ec2-user para ejecutar el comando como el usuario final
echo "Descargando Miniconda..."
sudo -u ec2-user wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /home/ec2-user/miniconda.sh

# 2. Instalar Miniconda en el directorio HOME del usuario, en modo batch (-b)
echo "Instalando Miniconda..."
sudo -u ec2-user bash /home/ec2-user/miniconda.sh -b -p /home/ec2-user/miniconda

# 3. Inicializar Conda para el shell bash del ec2-user
# Esto añade las líneas de inicialización al .bashrc
echo "Inicializando Conda..."
sudo -u ec2-user /home/ec2-user/miniconda/bin/conda init bash

# 4. Cargar el nuevo .bashrc en el shell actual
# Esto es crucial para que el comando 'conda' esté disponible inmediatamente
source /home/ec2-user/.bashrc

conda create --name myapp python=3.12.12
conda activate myapp

# Then install requirements without sudo
pip install -r /opt/app/Equipo-equipo-aims-22/src/requirements.txt


EOT
: var.user_data)

  tag_specifications {
    resource_type = "instance"
    tags = { Name = "tf-asg-instance" }
  }
}