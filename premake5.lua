project "AtlasGenerator"
kind "StaticLib"

language "C++"
cppdialect "C++17"

objdir "%{wks.location}/build/obj/%{cfg.buildcfg}/%{cfg.system}/%{cfg.architecture}/%{prj.name}"

targetdir "%{wks.location}/build/bin/%{cfg.buildcfg}/%{cfg.system}/%{cfg.architecture}/%{prj.name}"

files {
	"include/**.h",
	"include/**.hpp",
	"src/**.cpp"
}

includedirs {
	"include"
}

defines { "LIBNEST2D_GEOMETRIES_clipper", "LIBNEST2D_OPTIMIZER_nlopt" }

filter "system:windows"
defines { "_WINDOWS" }


filter "configurations:Debug"
links {
	"ThirdParty/lib/libnest2d/%{cfg.architecture}/%{cfg.system}/nloptd",
	"ThirdParty/lib/libnest2d/%{cfg.architecture}/%{cfg.system}/polyclippingd",
}
defines { "DEBUG" }
runtime "Debug"
symbols "on"

filter "configurations:Release"
links {
	"ThirdParty/lib/libnest2d/%{cfg.architecture}/%{cfg.system}/nlopt",
	"ThirdParty/lib/libnest2d/%{cfg.architecture}/%{cfg.system}/polyclipping",
}
defines { "NDEBUG" }
runtime "Release"
optimize "on"
