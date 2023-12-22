const express = require("express");
const { PrismaClient } = require("@prisma/client");
const bcrypt = require("bcrypt");
const jwt = require("jsonwebtoken");
const dotenv = require("dotenv");
const { google } = require("googleapis");
const fileUpload = require("express-fileupload");
const fs = require("fs");
const { Storage } = require("@google-cloud/storage");
const path = require ("path");
const axios = require("axios");

const pathKey = path.resolve(
  __dirname,
  "D:\\Bangkit\\api\\bangkit-api1.0.1\\express-prisma\\serviceaccountkey.json"
);
const gcs = new Storage({
  projectId: "apiecocycle",
  keyFilename: pathKey,
});

const bucketName = "upload-image-url";
const bucket = gcs.bucket(bucketName);

const app = express();
const PORT = process.env.PORT || 5000;
const prisma = new PrismaClient();
app.use(fileUpload());

app.get("/", (req, res) => {
  console.log("Response success");
  res.send("Response traffic 2 Success!");
});

const oauth2Client = new google.auth.OAuth2(
  process.env.GOOGLE_CLIENT_ID,
  process.env.GOOGLE_CLIENT_SECRET,
  "http://localhost:5000/auth/google/callback"
);

const scopes = [
  "https://www.googleapis.com/auth/userinfo.email",
  "https://www.googleapis.com/auth/userinfo.profile",
];

const authorizationUrl = oauth2Client.generateAuthUrl({
  access_type: "offline",
  scope: scopes,
  include_granted_scopes: true,
});

app.use(express.json());

const accessValidation = (req, res, next) => {
  const { authorization } = req.headers;

  if (!authorization) {
    return res.status(401).json({
      message: "Token diperlukan",
    });
  }

  const token = authorization.split(" ")[1];
  const secret = process.env.JWT_SECRET;

  try {
    const jwtDecode = jwt.verify(token, secret);
    req.userData = jwtDecode;
    next();
  } catch (error) {
    console.error("JWT Verification Error:", error);

    if (error.name === "TokenExpiredError") {
      return res.status(401).json({
        message: "Token expired",
      });
    } else {
      return res.status(401).json({
        message: "Unauthorized",
      });
    }
  }
};



app.get("/auth/google", (req, res) => {
  res.redirect(authorizationUrl);
});

app.get("/auth/google/callback", async (req, res) => {
  const { code } = req.query;

  const { tokens } = await oauth2Client.getToken(code);

  oauth2Client.setCredentials(tokens);

  const oauth2 = google.oauth2({
    auth: oauth2Client,
    version: "v2",
  });

  const { data } = await oauth2.userinfo.get();

  if (!data.email || !data.name) {
    return res.json({
      data: data,
    });
  }

  let user = await prisma.user.findUnique({
    where: {
      email: data.email,
    },
  });

  if (!user) {
    user = await prisma.user.create({
      data: {
        name: data.name,
        email: data.email,
      },
    });
  }

  const payload = {
    id: user.id,
    name: user.name,
  };

  const secret = process.env.JWT_SECRET;

  const expiresIn = 60 * 60 * 1;

  const token = jwt.sign(payload, secret, { expiresIn: expiresIn });

  return res.json({
    data: {
      id: user.id,
      name: user.name,
      token: token,
    },
  });
});

app.post("/register", async (req, res) => {
  try {
    const { name, email, password } = req.body;

    if (!name || !email || !password) {
      return res
        .status(400)
        .json({ error: "Name, email, and password are required fields" });
    }

    const hashedPassword = await bcrypt.hash(password, 10);

    const result = await prisma.user.create({
      data: {
        name,
        email,
        password: hashedPassword,
      },
    });

    res.json({
      message: "User created successfully",
      user: result,
    });
  } catch (error) {
    console.error("Error creating user:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});

app.post("/login", async (req, res) => {
  try {
    const { email, password } = req.body;

    if (!email) {
      return res.status(400).json({
        message: "Email is required",
      });
    }

    const user = await prisma.user.findUnique({
      where: {
        email: email,
      },
    });

    if (!user) {
      return res.status(404).json({
        message: "User not found",
      });
    }

    if (!user.password) {
      return res.status(404).json({
        message: "Password not set",
      });
    }

    const isPasswordValid = await bcrypt.compare(password, user.password);

    if (isPasswordValid) {
      const payload = {
        id: user.id,
        name: user.name,
        email: user.email,
      };

      const secret = process.env.JWT_SECRET || "your-secret-key";
      const expiresIn = 60 * 60 * 1;

      const token = jwt.sign(payload, secret, { expiresIn: expiresIn });

      return res.json({
        data: {
          id: user.id,
          name: user.name,
          email: user.email,
          token: token,
        },
      });
    } else {
      return res.status(403).json({
        message: "Wrong password",
      });
    }
  } catch (error) {
    console.error(error);
    return res.status(500).json({
      message: "Internal server error",
    });
  }
});

app.post("/users/validations", accessValidation, async (req, res, next) => {
  const { name, email } = req.body;

  const result = await prisma.user.create({
    data: {
      name: name,
      email: email,
    },
  });
  res.json({
    data: result,
    message: `User created`,
  });
});

app.get("/users", async (req, res) => {
  const result = await prisma.user.findMany({
    select: {
      id: true,
      name: true,
      email: true,
    },
  });
  res.json({
    data: result,
    message: "User list",
  });
});

app.get("/users/:id", async (req, res) => {
  const userId = parseInt(req.params.id);

  try {
    const user = await prisma.user.findUnique({
      where: { id: userId },
      select: {
        id: true,
        name: true,
        email: true,
      },
    });

    if (!user) {
      res.status(404).json({ error: "User not found" });
      return;
    }

    res.json({
      data: user,
      message: "User details",
    });
  } catch (error) {
    res.status(500).json({ error: "Internal Server Error" });
  }
});

app.get("/users", accessValidation, async (req, res) => {
  const result = await prisma.user.findMany({
    select: {
      id: true,
      name: true,
      email: true,
    },
  });
  res.json({
    data: result,
    message: "User list",
  });
});

app.patch("/users/:id", accessValidation, async (req, res) => {
  const { id } = req.params;
  const { name, email } = req.body;

  const result = await prisma.user.update({
    data: {
      name: name,
      email: email,
    },
    where: {
      id: Number(id),
    },
  });
  res.json({
    data: result,
    message: `User ${id} updated`,
  });
});

app.delete("/users/:id", accessValidation, async (req, res) => {
  const { id } = req.params;

  const result = await prisma.user.delete({
    where: {
      id: Number(id),
    },
  });
  res.json({
    message: `User ${id} deleted`,
  });
});

app.post("/postimage", accessValidation, async (req, res) => {
  try {
    const { files, userData } = req;

    if (!files || Object.keys(files).length === 0) {
      return res.status(400).json({ error: "No files were uploaded." });
    }

    const image = files.image;

    const currentDate = new Date();
    const year = currentDate.getFullYear();
    const month = String(currentDate.getMonth() + 1).padStart(2, "0");
    const day = String(currentDate.getDate()).padStart(2, "0");

    // Format the date
    const formattedDate = `${year}-${month}-${day}`;

    // Apply replacement of spaces, slashes, and commas with underscores
    const replacedDate = formattedDate.replace(/[\s\/,]+/g, "_");

    const dynamicUploadPath = `uploads/${replacedDate}/`;
    const uploadPath = __dirname + "/" + dynamicUploadPath + image.name;

    await fs.promises.mkdir(__dirname + "/" + dynamicUploadPath, {
      recursive: true,
    });

    image.mv(uploadPath, async (err) => {
      if (err) {
        return res.status(500).json({ error: err.message });
      }

      // Define destination here, based on your logic
      const destination = dynamicUploadPath + image.name; // Replace "some_value" with the actual destination value

      const gcsUploadOptions = {
        destination,
        metadata: {
          cacheControl: "public, max-age=31536000",
        },
      };

      await bucket.upload(uploadPath, gcsUploadOptions);

      const imageUrl = `https://storage.googleapis.com/${bucketName}/${destination}`;

      const result = await prisma.image.create({
        data: {
          path: imageUrl,
          userId: userData.id,
        },
      });

      res.json({
        message: "Image uploaded successfully",
        data: result,
      });
    });
  } catch (error) {
    console.error("Error uploading image:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});

app.get("/userimages", accessValidation, async (req, res) => {
  try {
    const { userData } = req;

    const userImages = await prisma.image.findMany({
      where: {
        userId: {
          equals: userData.id, // Menggunakan userId dari userData
        },
      },
      select: {
        id: true,
        path: true,
        userId: true,
      },
    });

    res.json({
      data: userImages,
      message: "User images retrieved successfully",
    });
  } catch (error) {
    console.error("Error retrieving user images:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});


app.post("/processingImg", accessValidation, async (req, res) => {
  try {
    const { path } = req.body; // Assuming the result path is provided in the request body

    if (!path) {
      return res
        .status(400)
        .json({ error: "Missing 'path' in the request body." });
    }

     const imageUrl = `https://storage.googleapis.com/${bucketName}/${destination}`;
    // Make a POST request to the provided URL
    const apiUrl =
      "https://asia-southeast2-apiecocycle.cloudfunctions.net/img_classifier_model";
    const response = await axios.post(apiUrl, { url: imageUrl });

    // Assuming the response from the external API is handled appropriately
    const apiResponse = response.data;

    res.json({
      message: "Image processing successful",
      apiResponse,
    });
  } catch (error) {
    console.error("Error processing image:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});

app.listen(PORT, () => {
  console.log(`Server running in PORT: ${PORT}`);
});
